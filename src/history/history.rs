#![allow(clippy::module_inception)]
use crate::cli::SortOrder;
use crate::history::{db_extensions, schema};
use crate::path_update_helpers;
use crate::fuzzy_matcher::SkimFuzzyMatcher;
use std::cell::RefCell;
use crate::settings::{HistoryFormat, ResultFilter, ResultSort, Settings, TimeRange};
use crate::shell_history;
use crate::simplified_command::SimplifiedCommand;
use crate::time::to_datetime;
use itertools::Itertools;
use rusqlite::named_params;
use rusqlite::types::ToSql;
use rusqlite::{Connection, MappedRows, Row};
use serde::{Serialize, Serializer};
use std::cmp::Ordering;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{fmt, fs, io};
use crate::ml::scalable_mlp::ScalableMlp;

#[derive(Debug, Clone, Default)]
pub struct Features {
    pub age_factor: f64,
    pub length_factor: f64,
    pub exit_factor: f64,
    pub recent_failure_factor: f64,
    pub selected_dir_factor: f64,
    pub dir_factor: f64,
    pub overlap_factor: f64,
    pub immediate_overlap_factor: f64,
    pub selected_occurrences_factor: f64,
    pub occurrences_factor: f64,
    // Enhanced match quality features from skim
    pub match_score: f64,          // Raw fuzzy match score
    pub match_positions: f64,      // Number of matched character positions
    pub match_density: f64,        // Ratio of matched chars to total chars
    pub match_gap_penalty: f64,    // Penalty for gaps between matches
    pub match_start_bonus: f64,    // Bonus for matches at word/command start
    pub match_span_ratio: f64,     // Ratio of match span to total length
}

#[derive(Debug, Clone, Default)]
pub struct Command {
    pub id: i64,
    pub cmd: String,
    pub cmd_tpl: String,
    pub session_id: String,
    pub rank: f64,
    pub when_run: Option<i64>,
    pub last_run: Option<i64>,
    pub exit_code: Option<i32>,
    pub selected: bool,
    pub dir: Option<String>,
    pub features: Features,
    pub match_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DumpCommand {
    pub id: i64,
    pub cmd: String,
    pub cmd_tpl: String,
    pub session_id: String,
    #[serde(serialize_with = "ser_to_datetime")]
    pub when_run: i64,
    pub exit_code: i32,
    pub selected: i32,
    pub dir: Option<String>,
    pub old_dir: Option<String>,
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.cmd.fmt(f)
    }
}

impl From<Command> for String {
    fn from(command: Command) -> Self {
        command.cmd
    }
}

#[inline]
fn ser_to_datetime<S>(when_run: &i64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&to_datetime(*when_run))
}

pub struct History {
    pub connection: Connection,
    matcher: RefCell<Option<SkimFuzzyMatcher>>,
    learner: RefCell<Option<ScalableMlp>>,
    // Controls how often the model is saved to disk (1 = every update).
    save_frequency: u32,
    // Counts updates since load; used to decide when to persist.
    update_count: RefCell<u32>,
}

impl std::fmt::Debug for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("History")
            .field("connection", &"<Connection>")
            // network removed
            .field("matcher", &"<Matcher>")
            .finish()
    }
}

const IGNORED_COMMANDS: [&str; 7] = [
    "pwd",
    "ls",
    "cd",
    "cd ..",
    "clear",
    "history",
    "mcfly search",
];

impl History {
    /// Lazily initialize and return a mutable borrow to the shared SkimFuzzyMatcher.
    fn matcher_mut(&self) -> std::cell::RefMut<SkimFuzzyMatcher> {
        // If not initialized yet, create the SkimFuzzyMatcher.
        if self.matcher.borrow().is_none() {
            *self.matcher.borrow_mut() = Some(SkimFuzzyMatcher::new());
        }
        std::cell::RefMut::map(self.matcher.borrow_mut(), |opt: &mut Option<SkimFuzzyMatcher>| opt.as_mut().unwrap())
    }

    #[must_use]
    pub fn load(history_format: HistoryFormat, save_frequency: u32) -> History {
        let db_path = Settings::mcfly_db_path();
        let history = if db_path.exists() {
            History::from_db_path(db_path)
        } else {
            History::from_shell_history(history_format)
        };
        schema::migrate(&history.connection);
        // Try to load learner model (which now embeds update_count) if present
        let (learner, starting_update_count) = match Settings::mcfly_db_path().parent() {
            Some(p) => {
                let model_path = p.join("scalable-ml.yaml");
                match ScalableMlp::load(&model_path) {
                    Some(m) => {
                        let uc = m.update_count;
                        (Some(m), uc)
                    }
                    None => (None, 0usize),
                }
            }
            None => (None, 0usize),
        };

        History {
            learner: RefCell::new(learner),
            save_frequency,
            update_count: RefCell::new(starting_update_count as u32),
            ..history
        }
    }

    /// Score features and optionally update the model on demand.
    /// `update` controls whether a single SGD step is applied with `target` and `lr`.
    pub fn score_and_maybe_update(&self, features: &[f64], update: bool, target: Option<&[f64]>, lr: f64, weight_decay: f64, _momentum: f64, _optimizer: &str) -> Vec<f64> {
        // Ensure a learner exists
        if self.learner.borrow().is_none() {
            // Create a 3-layer network: 16 input features -> 32 -> 16 -> 1 output
            let model = ScalableMlp::new(16, 32, 16, 1);
            *self.learner.borrow_mut() = Some(model);
        }

        let mut learner = self.learner.borrow_mut();
        let learner_ref = learner.as_mut().unwrap();
        let scores = learner_ref.score(features);
        if update {
            if let Some(t) = target {
                learner_ref.update(features, t, lr, weight_decay);
                // persist model according to save_frequency
                let mut count = self.update_count.borrow_mut();
                *count = count.wrapping_add(1);
                if *count % self.save_frequency == 0 {
                    if let Some(p) = Settings::mcfly_db_path().parent() {
                        let model_path = p.join("scalable-ml.yaml");
                        // store the count inside the model YAML before saving
                        learner_ref.update_count = *count as usize;
                        let _ = learner_ref.save(&model_path);
                    }
                }
            }
        }

        scores
    }

    pub fn should_add(&self, command: &str) -> bool {
        // Ignore empty commands.
        if command.is_empty() {
            return false;
        }

        // Ignore commands added via a ctrl-r search.
        if command.starts_with("#mcfly:") {
            return false;
        }

        // Ignore commands with a leading space.
        if command.starts_with(' ') {
            return false;
        }

        // Ignore blacklisted commands.
        if IGNORED_COMMANDS.contains(&command) {
            return false;
        }

        // Ignore the previous command (independent of Session ID) so that opening a new terminal
        // window won't replay the last command in the history.
        let last_command = self.last_command(&None);
        if last_command.is_none() {
            return true;
        }
        !command.eq(&last_command.unwrap().cmd)
    }

    pub fn add(
        &self,
        command: &str,
        session_id: &str,
        dir: &str,
        when_run: &Option<i64>,
        exit_code: Option<i32>,
        old_dir: &Option<String>,
    ) {
        self.possibly_update_paths(command, exit_code);
        let selected = self.determine_if_selected_from_ui(command, session_id, dir);
        let simplified_command = SimplifiedCommand::new(command, true);
        self.connection.execute("INSERT INTO commands (cmd, cmd_tpl, session_id, when_run, exit_code, selected, dir, old_dir) VALUES (:cmd, :cmd_tpl, :session_id, :when_run, :exit_code, :selected, :dir, :old_dir)",
                                named_params!{
                                    ":cmd": &command.to_owned(),
                                    ":cmd_tpl": &simplified_command.result,
                                    ":session_id": &session_id.to_owned(),
                                    ":when_run": &when_run.to_owned(),
                                    ":exit_code": &exit_code.clone(),
                                    ":selected": &selected,
                                    ":dir": &dir.to_owned(),
                                    ":old_dir": &old_dir.to_owned(),
                                }).unwrap_or_else(|err| panic!("McFly error: Insert into commands to work ({err})"));
    }

    fn determine_if_selected_from_ui(&self, command: &str, session_id: &str, dir: &str) -> bool {
        let rows_affected = self
            .connection
            .execute(
                "DELETE FROM selected_commands \
                 WHERE cmd = :cmd \
                 AND session_id = :session_id \
                 AND dir = :dir",
                &[
                    (":cmd", &command.to_owned()),
                    (":session_id", &session_id.to_owned()),
                    (":dir", &dir.to_owned()),
                ],
            )
            .unwrap_or_else(|err| {
                panic!("McFly error: DELETE from selected_commands to work ({err})")
            });

        // Delete any other pending selected commands for this session -- they must have been aborted or edited.
        self.connection
            .execute(
                "DELETE FROM selected_commands WHERE session_id = :session_id",
                &[(":session_id", &session_id.to_owned())],
            )
            .unwrap_or_else(|err| {
                panic!("McFly error: DELETE from selected_commands to work ({err})")
            });

        rows_affected > 0
    }

    pub fn record_selected_from_ui(&self, command: &str, session_id: &str, dir: &str) {
        self.connection.execute("INSERT INTO selected_commands (cmd, session_id, dir) VALUES (:cmd, :session_id, :dir)",
                                      &[
                                          (":cmd", &command.to_owned()),
                                          (":session_id", &session_id.to_owned()),
                                          (":dir", &dir.to_owned())
                                      ]).unwrap_or_else(|err| panic!("McFly error: Insert into selected_commands to work ({err})"));
    }

    // Update historical paths in our database if a directory has been renamed or moved.
    pub fn possibly_update_paths(&self, command: &str, exit_code: Option<i32>) {
        let successful = exit_code.is_none() || exit_code.unwrap() == 0;
        let is_move =
            |c: &str| c.to_lowercase().starts_with("mv ") && !c.contains('*') && !c.contains('?');
        if successful && is_move(command) {
            let parts = path_update_helpers::parse_mv_command(command);
            if parts.len() == 2 {
                let normalized_from = path_update_helpers::normalize_path(&parts[0]);
                let normalized_to = path_update_helpers::normalize_path(&parts[1]);

                // If $to/$(base_name($from)) exists, and is a directory, assume we've moved $from into $to.
                // If not, assume we've renamed $from to $to.

                if let Some(basename) = PathBuf::from(&normalized_from).file_name() {
                    if let Some(utf8_basename) = basename.to_str() {
                        if utf8_basename.contains('.') {
                            // It was probably a file.
                            return;
                        }
                        let maybe_moved_directory =
                            PathBuf::from(&normalized_to).join(utf8_basename);
                        if maybe_moved_directory.exists() {
                            if maybe_moved_directory.is_dir() {
                                self.update_paths(
                                    &normalized_from,
                                    maybe_moved_directory.to_str().unwrap(),
                                    false,
                                );
                            } else {
                                // The source must have been a file, so ignore it.
                            }
                            return;
                        }
                    } else {
                        // Don't try to handle non-utf8 filenames, at least for now.
                        return;
                    }
                }

                let to_pathbuf = PathBuf::from(&normalized_to);
                if to_pathbuf.exists() && to_pathbuf.is_dir() {
                    self.update_paths(&normalized_from, &normalized_to, false);
                }
            }
        }
    }

    pub fn find_matches(
        &self,
        cmd: &str,
        num: i16,
        fuzzy: i16,
        result_sort: &ResultSort,
    ) -> Vec<Command> {
        let (wildcard, match_function, cmd) = if Self::is_case_sensitive(cmd) {
            // escape '*' with '[*]' and replace '%' with '*' for glob matching
            ("*", "GLOB", cmd.replace("*", "[*]").replace("%", "*"))
        } else {
            ("%", "LIKE", cmd.to_string())
        };

        let mut like_query = wildcard.to_string();

        if fuzzy > 0 {
            like_query.push_str(&cmd.split("").collect::<Vec<&str>>().join(wildcard));
        } else {
            like_query.push_str(&cmd);
        }

        like_query += wildcard;

        let order_by_column: &str = match &result_sort {
            ResultSort::LastRun => "last_run",
            _ => "rank",
        };

        let query: &str = &format!(
            "{} {} {} {} {}",
            "SELECT id, cmd, cmd_tpl, session_id, when_run, exit_code, selected, dir, rank,
                age_factor, length_factor, exit_factor, recent_failure_factor,
                selected_dir_factor, dir_factor, overlap_factor, immediate_overlap_factor,
                selected_occurrences_factor, occurrences_factor, last_run
            FROM contextual_commands
            WHERE cmd",
            match_function,
            "(:like)
            ORDER BY",
            order_by_column,
            "DESC LIMIT :limit"
        )[..];

        let mut statement = self
            .connection
            .prepare(query)
            .unwrap_or_else(|err| panic!("McFly error: Prepare to work ({err})"));
        let command_iter = statement
            .query_map(
                named_params! { ":like": &like_query, ":limit": &num },
                |row| {
                    let text: String = row
                        .get(1)
                        .unwrap_or_else(|err| panic!("McFly error: cmd to be readable ({err})"));

                    let (bounds, match_features) = self.calc_match_indices_with_features(&text, &cmd, fuzzy);

                    Ok(Command {
                        id: row.get(0).unwrap_or_else(|err| {
                            panic!("McFly error: id to be readable ({err})")
                        }),
                        cmd: text,
                        cmd_tpl: row.get(2).unwrap_or_else(|err| {
                            panic!("McFly error: cmd_tpl to be readable ({err})")
                        }),
                        session_id: row.get(3).unwrap_or_else(|err| {
                            panic!("McFly error: session_id to be readable ({err})")
                        }),
                        when_run: row.get(4).unwrap_or_else(|err| {
                            panic!("McFly error: when_run to be readable ({err})")
                        }),
                        exit_code: row.get(5).unwrap_or_else(|err| {
                            panic!("McFly error: exit_code to be readable ({err})")
                        }),
                        selected: row.get(6).unwrap_or_else(|err| {
                            panic!("McFly error: selected to be readable ({err})")
                        }),
                        dir: row.get(7).unwrap_or_else(|err| {
                            panic!("McFly error: dir to be readable ({err})")
                        }),
                        rank: row.get(8).unwrap_or_else(|err| {
                            panic!("McFly error: rank to be readable ({err})")
                        }),
                        match_indices: bounds,
                        features: Features {
                            age_factor: row.get(9).unwrap_or_else(|err| {
                                panic!("McFly error: age_factor to be readable ({err})")
                            }),
                            length_factor: row.get(10).unwrap_or_else(|err| {
                                panic!("McFly error: length_factor to be readable ({err})")
                            }),
                            exit_factor: row.get(11).unwrap_or_else(|err| {
                                panic!("McFly error: exit_factor to be readable ({err})")
                            }),
                            recent_failure_factor: row.get(12).unwrap_or_else(|err| {
                                panic!(
                                    "McFly error: recent_failure_factor to be readable ({err})"
                                )
                            }),
                            selected_dir_factor: row.get(13).unwrap_or_else(|err| {
                                panic!("McFly error: selected_dir_factor to be readable ({err})")
                            }),
                            dir_factor: row.get(14).unwrap_or_else(|err| {
                                panic!("McFly error: dir_factor to be readable ({err})")
                            }),
                            overlap_factor: row.get(15).unwrap_or_else(|err| {
                                panic!("McFly error: overlap_factor to be readable ({err})")
                            }),
                            immediate_overlap_factor: row.get(16).unwrap_or_else(|err| {
                                panic!(
                                    "McFly error: immediate_overlap_factor to be readable ({err})"
                                )
                            }),
                            selected_occurrences_factor: row.get(17).unwrap_or_else(|err| {
                                panic!(
                                    "McFly error: selected_occurrences_factor to be readable ({err})"
                                )
                            }),
                            occurrences_factor: row.get(18).unwrap_or_else(|err| {
                                panic!("McFly error: occurrences_factor to be readable ({err})")
                            }),
                            // Enhanced match quality features from skim
                            match_score: match_features.0,
                            match_positions: match_features.1,
                            match_density: match_features.2,
                            match_gap_penalty: match_features.3,
                            match_start_bonus: match_features.4,
                            match_span_ratio: match_features.5,
                        },
                        last_run: row.get(19).unwrap_or_else(|err| {
                            panic!("McFly error: last_run to be readable ({err})")
                        }),
                    })
                },
            )
            .unwrap_or_else(|err| panic!("McFly error: Query Map to work ({err})"));

        let mut names = Vec::new();
        for result in command_iter {
            names.push(result.unwrap_or_else(|err| {
                panic!("McFly error: Unable to load command from DB ({err})")
            }));
        }

        if fuzzy > 0 && result_sort != &ResultSort::LastRun {
            names = names
                .into_iter()
                .sorted_unstable_by(|a, b| {
                    // Fuzzy matches impose new ordering criteria on top of the
                    // natural rank-based sorting: at the most basic level,
                    // shorter and earlier matches are more likely to be
                    // desired than longer or later matches -- even if they are
                    // ranked a little lower.
                    //
                    // Each match is weighted by the inverse of its length plus
                    // start position, relative to the total length + start of
                    // both matches added together. This yields two complements
                    // which always add up to 1 (e.g. 0.6 vs 0.4). If both
                    // matches have the same length and start position, or if
                    // those balance out exactly, the resulting weights will
                    // both equal 0.5.
                    //
                    // The weights are multiplied by the configurable fuzzy
                    // factor before being added to each result's original
                    // rank. Factors > 1 are a "thumb on the scale" increasing
                    // the likelihood of the weight flipping the outcome for
                    // the originally lower-ranked result.

                    let a_start = *a.match_indices.first().unwrap_or(&0);
                    let b_start = *b.match_indices.first().unwrap_or(&0);

                    let a_len = a.match_indices.last().map(|i| i + 1).unwrap_or(0) - a_start;
                    let b_len = b.match_indices.last().map(|i| i + 1).unwrap_or(0) - b_start;

                    let a_mod =
                        1.0 - (a_start + a_len) as f64 / (a_start + b_start + a_len + b_len) as f64;
                    let b_mod =
                        1.0 - (b_start + b_len) as f64 / (a_start + b_start + a_len + b_len) as f64;

                    PartialOrd::partial_cmp(
                        &(b.rank + b_mod * f64::from(fuzzy)),
                        &(a.rank + a_mod * f64::from(fuzzy)),
                    )
                    .unwrap_or(Ordering::Equal)
                })
                .collect();
        }

        names
    }

    /// Enable case sensitivity when input string contains uppercase
    fn is_case_sensitive(cmd: &str) -> bool {
        cmd.chars().any(|c| c.is_uppercase())
    }

    /// Calculate the indices of the matches in the text and extract match quality features.
    /// Returns (match_indices, match_features) where match_features contains enhanced scoring data.
    fn calc_match_indices_with_features(&self, text: &str, cmd: &str, fuzzy: i16) -> (Vec<usize>, (f64, f64, f64, f64, f64, f64)) {
        let (text_s, cmd_s) = if Self::is_case_sensitive(cmd) {
            (text.to_string(), cmd.to_string())
        } else {
            (text.to_lowercase(), cmd.to_lowercase())
        };

        match fuzzy {
            0 => {
                // Exact match
                let matches: Vec<_> = text_s
                    .match_indices(&cmd_s)
                    .flat_map(|(index, _)| index..index + cmd_s.len())
                    .collect();
                let match_features = if matches.is_empty() {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                } else {
                    // For exact matches, provide perfect scores
                    let match_count = cmd_s.len() as f64;
                    let density = match_count / text_s.len() as f64;
                    let start_bonus = if matches.first().unwrap_or(&usize::MAX) == &0 { 1.0 } else { 0.0 };
                    (100.0, match_count, density, 0.0, start_bonus, density)
                };
                (matches, match_features)
            }
            _ => {
                let mut matcher = self.matcher_mut();
                if let Some((indices, score)) = matcher.match_indices(&text_s, &cmd_s) {
                    // Calculate enhanced match quality features
                    let match_positions = indices.len() as f64;
                    let text_len = text_s.len() as f64;
                    let _cmd_len = cmd_s.len() as f64;
                    let match_density = match_positions / text_len;
                    
                    // Calculate gap penalty (average gap size between consecutive matches)
                    let gap_penalty = if indices.len() > 1 {
                        let total_gaps: usize = indices.windows(2).map(|w| w[1] - w[0] - 1).sum();
                        total_gaps as f64 / (indices.len() - 1) as f64
                    } else {
                        0.0
                    };
                    
                    // Start bonus: higher score if match starts early
                    let start_bonus = if let Some(&first_idx) = indices.first() {
                        1.0 - (first_idx as f64 / text_len)
                    } else {
                        0.0
                    };
                    
                    // Match span ratio: how much of the text is covered by the match
                    let span_ratio = if let (Some(&first), Some(&last)) = (indices.first(), indices.last()) {
                        (last - first + 1) as f64 / text_len
                    } else {
                        0.0
                    };
                    
                    let normalized_score = score as f64; // Use raw skim score
                    let match_features = (normalized_score, match_positions, match_density, gap_penalty, start_bonus, span_ratio);
                    (indices, match_features)
                } else {
                    (Vec::new(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                }
            }
        }
    }

    /// Legacy method for backwards compatibility - only returns indices
    #[allow(dead_code)]
    fn calc_match_indices(&self, text: &str, cmd: &str, fuzzy: i16) -> Vec<usize> {
        self.calc_match_indices_with_features(text, cmd, fuzzy).0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_cache_table(
        &self,
        dir: &str,
        result_filter: &ResultFilter,
        session_id: &Option<String>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        now: Option<i64>,
        limit: Option<i64>,
    ) {
        let lookback: u16 = 3;

        let mut last_commands = self.last_command_templates(session_id, lookback as i16, 0);
        if last_commands.len() < lookback as usize {
            last_commands = self.last_command_templates(&None, lookback as i16, 0);
            while last_commands.len() < lookback as usize {
                last_commands.push(String::new());
            }
        }

        #[allow(unused_variables)]
        let beginning_of_execution = Instant::now();

        self.connection
            .execute("PRAGMA temp_store = MEMORY;", [])
            .unwrap();

        self.connection
            .execute("DROP TABLE IF EXISTS temp.contextual_commands;", [])
            .unwrap_or_else(|err| panic!("McFly error: Removal of temp table to work ({err})"));

        let (mut when_run_min, when_run_max): (f64, f64) = self
            .connection
            .query_row(
                "SELECT IFNULL(MIN(when_run), 0), IFNULL(MAX(when_run), 0) FROM commands",
                [],
                |row| Ok((row.get_unwrap(0), row.get_unwrap(1))),
            )
            .unwrap_or_else(|err| panic!("McFly error: Query to work ({err})"));

        if (when_run_min - when_run_max).abs() < f64::EPSILON {
            when_run_min -= 60.0 * 60.0;
        }

        let max_occurrences: f64 = self
            .connection
            .query_row(
                "SELECT COUNT(*) AS c FROM commands GROUP BY cmd ORDER BY c DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(1.0);

        let max_selected_occurrences: f64 = self.connection
            .query_row("SELECT COUNT(*) AS c FROM commands WHERE selected = 1 GROUP BY cmd ORDER BY c DESC LIMIT 1", [],
                       |row| row.get(0)).unwrap_or(1.0); // FIXME: 1.0 seems wrong.

        let max_length: f64 = self
            .connection
            .query_row(
                "SELECT IFNULL(MAX(LENGTH(cmd)), 0) FROM commands",
                [],
                |row| row.get(0),
            )
            .unwrap_or(100.0);

        let max_id: i64 = self
            .connection
            .query_row("SELECT IFNULL(MAX(id), 0) FROM commands", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        let min_id = if let Some(limit_value) = limit {
            if limit_value > max_id {
                0
            } else {
                (max_id as f64 * (1.0 - (limit_value as f64 / max_id as f64))) as i64
            }
        } else {
            0
        };

        let dir_filter_off = match &result_filter {
            ResultFilter::Global => true,
            ResultFilter::CurrentDirectory => false,
        };

        self.connection.execute(
            "CREATE TEMP TABLE contextual_commands AS SELECT
                  id, cmd, cmd_tpl, session_id, when_run, MAX(when_run) AS last_run, exit_code, selected, dir,

                  /* to be filled in later */
                  0.0 AS rank,

                  /* length of the command string */
                  LENGTH(c.cmd) / :max_length AS length_factor,

                  /* age of the last execution of this command (0.0 is new, 1.0 is old) */
                  MIN((:when_run_max - when_run) / :history_duration) AS age_factor,

                  /* average error state (1: always successful, 0: always errors) */
                  SUM(CASE WHEN exit_code = 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) as exit_factor,

                  /* recent failure (1 if failed recently, 0 if not) */
                  MAX(CASE WHEN exit_code != 0 AND :now - when_run < 120 THEN 1.0 ELSE 0.0 END) AS recent_failure_factor,

                  /* percentage run in this directory (1: always run in this directory, 0: never run in this directory) */
                  SUM(CASE WHEN dir = :directory THEN 1.0 ELSE 0.0 END) / COUNT(*) as dir_factor,

                  /* percentage of time selected in this directory (1: only selected in this dir, 0: only selected elsewhere) */
                  SUM(CASE WHEN dir = :directory AND selected = 1 THEN 1.0 ELSE 0.0 END) / (SUM(CASE WHEN selected = 1 THEN 1.0 ELSE 0.0 END) + 1) as selected_dir_factor,

                  /* average contextual overlap of this command (0: none of the last 3 commands has ever overlapped with this command, 1: all of the last three commands always overlap with this command) */
                  SUM((
                    SELECT COUNT(DISTINCT c2.cmd_tpl) FROM commands c2
                    WHERE c2.id >= c.id - :lookback AND c2.id < c.id AND c2.cmd_tpl IN (:last_commands0, :last_commands1, :last_commands2)
                  ) / :lookback_f64) / COUNT(*) AS overlap_factor,

                  /* average overlap with the last command (0: this command never follows the last command, 1: this command always follows the last command) */
                  SUM((SELECT COUNT(*) FROM commands c2 WHERE c2.id = c.id - 1 AND c2.cmd_tpl = :last_commands0)) / COUNT(*) AS immediate_overlap_factor,

                  /* percentage selected (1: this is the most commonly selected command, 0: this command is never selected) */
                  SUM(CASE WHEN selected = 1 THEN 1.0 ELSE 0.0 END) / :max_selected_occurrences AS selected_occurrences_factor,

                  /* percentage of time this command is run relative to the most common command (1: this is the most common command, 0: this is the least common command) */
                  COUNT(*) / :max_occurrences AS occurrences_factor

                  FROM commands c
                  WHERE id > :min_id AND when_run > :start_time AND when_run < :end_time AND (:dir_filter_off OR dir LIKE :directory)
                  GROUP BY cmd
                  ORDER BY id DESC;",
            named_params! {
                ":when_run_max": &when_run_max,
                ":history_duration": &(when_run_max - when_run_min),
                ":directory": &dir.to_owned(),
                ":dir_filter_off": &dir_filter_off,
                ":max_occurrences": &max_occurrences,
                ":max_length": &max_length,
                ":max_selected_occurrences": &max_selected_occurrences,
                ":lookback": &lookback,
                ":lookback_f64": &f64::from(lookback),
                ":last_commands0": &last_commands[0].clone(),
                ":last_commands1": &last_commands[1].clone(),
                ":last_commands2": &last_commands[2].clone(),
                ":start_time": &start_time.unwrap_or(0).to_owned(),
                ":end_time": &end_time.unwrap_or(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_else(|err| panic!("McFly error: Time went backwards ({err})")).as_secs() as i64).to_owned(),
                ":now": &now.unwrap_or(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_else(|err| panic!("McFly error: Time went backwards ({err})")).as_secs() as i64).to_owned(),
                ":min_id": &min_id,
            }).unwrap_or_else(|err| panic!("McFly error: Creation of temp table to work ({err})"));

        self.connection
            .execute(
                "UPDATE contextual_commands
                 SET rank = nn_rank(age_factor, length_factor, exit_factor,
                                    recent_failure_factor, selected_dir_factor, dir_factor,
                                    overlap_factor, immediate_overlap_factor,
                                    selected_occurrences_factor, occurrences_factor);",
                [],
            )
            .unwrap_or_else(|err| panic!("McFly error: Ranking of temp table to work ({err})"));

        self.connection
            .execute("CREATE INDEX temp.MyIndex ON contextual_commands(id);", [])
            .unwrap_or_else(|err| {
                panic!("McFly error: Creation of index on temp table to work ({err})")
            });

        // println!("Seconds: {}", (beginning_of_execution.elapsed().as_secs() as f64) + (beginning_of_execution.elapsed().subsec_nanos() as f64 / 1000_000_000.0));
    }

    pub fn commands(
        &self,
        session_id: &Option<String>,
        num: i16,
        offset: u16,
        random: bool,
    ) -> Vec<Command> {
        let order = if random { "RANDOM()" } else { "id" };
        let query = if session_id.is_none() {
            format!("SELECT id, cmd, cmd_tpl, session_id, when_run, exit_code, selected, dir FROM commands ORDER BY {order} DESC LIMIT :limit OFFSET :offset")
        } else {
            format!("SELECT id, cmd, cmd_tpl, session_id, when_run, exit_code, selected, dir FROM commands WHERE session_id = :session_id ORDER BY {order} DESC LIMIT :limit OFFSET :offset")
        };

        let closure: fn(&Row) -> rusqlite::Result<Command> = |row| {
            Ok(Command {
                id: row.get(0)?,
                cmd: row.get(1)?,
                cmd_tpl: row.get(2)?,
                session_id: row.get(3)?,
                when_run: row.get(4)?,
                exit_code: row.get(5)?,
                selected: row.get(6)?,
                dir: row.get(7)?,
                ..Command::default()
            })
        };

        if session_id.is_none() {
            self.run_query(&query, &[(":limit", &num), (":offset", &offset)], closure)
        } else {
            self.run_query(
                &query,
                &[
                    (":session_id", &session_id.to_owned().unwrap()),
                    (":limit", &num),
                    (":offset", &offset),
                ],
                closure,
            )
        }
    }

    pub fn run_query<T, F>(&self, query: &str, params: &[(&str, &dyn ToSql)], f: F) -> Vec<T>
    where
        F: FnMut(&Row<'_>) -> rusqlite::Result<T>,
    {
        let mut statement = self.connection.prepare(query).unwrap();

        let rows: MappedRows<_> = statement
            .query_map(params, f)
            .unwrap_or_else(|err| panic!("McFly error: Query Map to work ({err})"));

        let mut vec: Vec<T> = Vec::new();
        for row in rows.flatten() {
            vec.push(row);
        }

        vec
    }

    pub fn last_command(&self, session_id: &Option<String>) -> Option<Command> {
        self.commands(session_id, 1, 0, false).first().cloned()
    }

    pub fn last_command_templates(
        &self,
        session_id: &Option<String>,
        num: i16,
        offset: u16,
    ) -> Vec<String> {
        self.commands(session_id, num, offset, false)
            .iter()
            .map(|command| command.cmd_tpl.clone())
            .collect()
    }

    pub fn delete_command(&self, command: &str) {
        self.connection
            .execute(
                "DELETE FROM selected_commands WHERE cmd = :command",
                &[(":command", &command)],
            )
            .unwrap_or_else(|err| {
                panic!("McFly error: DELETE from selected_commands to work ({err})")
            });

        self.connection
            .execute(
                "DELETE FROM commands WHERE cmd = :command",
                &[(":command", &command)],
            )
            .unwrap_or_else(|err| panic!("McFly error: DELETE from commands to work ({err})"));
    }

    pub fn update_paths(&self, old_path: &str, new_path: &str, print_output: bool) {
        let normalized_old_path = path_update_helpers::normalize_path(old_path);
        let normalized_new_path = path_update_helpers::normalize_path(new_path);

        if normalized_old_path.len() > 1 && normalized_new_path.len() > 1 {
            let like_query = normalized_old_path.to_string() + "/%";

            let mut dir_update_statement = self.connection.prepare(
                "UPDATE commands SET dir = :new_dir || SUBSTR(dir, :length) WHERE dir = :exact OR dir LIKE (:like)"
            ).unwrap();

            let mut old_dir_update_statement = self.connection.prepare(
                "UPDATE commands SET old_dir = :new_dir || SUBSTR(old_dir, :length) WHERE old_dir = :exact OR old_dir LIKE (:like)"
            ).unwrap();

            let affected = dir_update_statement
                .execute(named_params! {
                       ":like": &like_query,
                       ":exact": &normalized_old_path,
                       ":new_dir": &normalized_new_path,
                       ":length": &(normalized_old_path.chars().count() as u32 + 1),
                })
                .unwrap_or_else(|err| panic!("McFly error: dir UPDATE to work ({err})"));

            old_dir_update_statement
                .execute(named_params! {
                    ":like": &like_query,
                    ":exact": &normalized_old_path,
                    ":new_dir": &normalized_new_path,
                    ":length": &(normalized_old_path.chars().count() as u32 + 1),
                })
                .unwrap_or_else(|err| panic!("McFly error: old_dir UPDATE to work ({err})"));

            if print_output {
                println!(
                    "McFly: Command database paths renamed from {normalized_old_path} to {normalized_new_path} (affected {affected} commands)"
                );
            }
        } else if print_output {
            println!("McFly: Not updating paths due to invalid options.");
        }
    }

    pub fn dump(&self, time_range: &TimeRange, order: &SortOrder) -> Vec<DumpCommand> {
        let mut where_clause = String::new();
        // Were there condtions in where clause?
        let mut has_conds = false;
        let mut params: Vec<(&str, &dyn ToSql)> = Vec::with_capacity(2);

        if !time_range.is_full() {
            where_clause.push_str("WHERE");

            if let Some(since) = &time_range.since {
                where_clause.push_str(" :since <= when_run");
                has_conds = true;
                params.push( (":since", since) );
            }

            if let Some(before) = &time_range.before {
                if has_conds {
                    where_clause.push_str(" AND");
                }
                where_clause.push_str(" when_run < :before");
                params.push( (":before", before) );
            }
        }

        let query = format!(
            "SELECT id, cmd, cmd_tpl, session_id, when_run, exit_code, selected, dir, old_dir FROM commands {} ORDER BY when_run {}",
            where_clause,
            order.to_str()
        );
        self.run_query(&query, params.as_slice(), |row| {
            Ok(DumpCommand {
                id: row.get(0)?,
                cmd: row.get(1)?,
                cmd_tpl: row.get(2)?,
                session_id: row.get(3)?,
                when_run: row.get(4)?,
                exit_code: row.get(5)?,
                selected: row.get(6)?,
                dir: row.get(7).ok(),
                old_dir: row.get(8).ok(),
            })
        })
    }

    fn from_shell_history(history_format: HistoryFormat) -> History {
        print!(
            "McFly: Importing shell history for the first time. This may take a minute or two..."
        );
        io::stdout()
            .flush()
            .unwrap_or_else(|err| panic!("McFly error: STDOUT flush should work ({err})"));

        // Load this first to make sure it works before we create the DB.
        let commands =
            shell_history::full_history(&shell_history::history_file_path(), history_format);

        // Use ~/.mcfly if it already exists, or create 'mcfly' folder in XDG_DATA_DIR
        let mcfly_db_path = Settings::mcfly_db_path();
        let mcfly_db_dir = mcfly_db_path.parent().unwrap();

        fs::create_dir_all(mcfly_db_dir)
            .unwrap_or_else(|_| panic!("Unable to create {mcfly_db_dir:?}"));

        // Make ~/.mcfly/history.db
        let mut connection = Connection::open(&mcfly_db_path)
            .unwrap_or_else(|_| panic!("Unable to create history DB at {:?}", &mcfly_db_path));

        db_extensions::add_db_functions(&connection);

        connection.execute_batch(
            "CREATE TABLE commands( \
                      id INTEGER PRIMARY KEY AUTOINCREMENT, \
                      cmd TEXT NOT NULL, \
                      cmd_tpl TEXT, \
                      session_id TEXT NOT NULL, \
                      when_run INTEGER NOT NULL, \
                      exit_code INTEGER NOT NULL, \
                      selected INTEGER NOT NULL, \
                      dir TEXT, \
                      old_dir TEXT \
                  ); \
                  CREATE INDEX command_cmds ON commands (cmd);\
                  CREATE INDEX command_session_id ON commands (session_id);\
                  CREATE INDEX command_dirs ON commands (dir);\
                  \
                  CREATE TABLE selected_commands( \
                      id INTEGER PRIMARY KEY AUTOINCREMENT, \
                      cmd TEXT NOT NULL, \
                      session_id TEXT NOT NULL, \
                      dir TEXT NOT NULL \
                  ); \
                  CREATE INDEX selected_command_session_cmds ON selected_commands (session_id, cmd);"
        ).unwrap_or_else(|err| panic!("McFly error: Unable to initialize history db ({err})"));

        let transaction = connection
            .transaction()
            .unwrap_or_else(|err| panic!("McFly error: Unable to begin transaction ({err})"));
        {
            let mut statement = transaction
                .prepare("INSERT INTO commands (cmd, cmd_tpl, session_id, when_run, exit_code, selected) VALUES (:cmd, :cmd_tpl, :session_id, :when_run, :exit_code, :selected)")
                .unwrap_or_else(|err| panic!("McFly error: Unable to prepare insert ({err})"));
            for command in commands {
                if !IGNORED_COMMANDS.contains(&command.command.as_str()) {
                    let simplified_command = SimplifiedCommand::new(&command.command, true);
                    if !command.command.is_empty() && !simplified_command.result.is_empty() {
                        if let Err(e) = statement.execute(named_params! {
                            ":cmd": &command.command,
                            ":cmd_tpl": &simplified_command.result.clone(),
                            ":session_id": &"IMPORTED",
                            ":when_run": &command.when,
                            ":exit_code": &0,
                            ":selected": &0,
                        }) {
                            println!(
                                "A single history line could not be saved due to '{}' (command was '{}'), but other inserts should be fine.",
                                e, &command.command
                            );
                        }
                    }
                }
            }
        }
        transaction
            .commit()
            .unwrap_or_else(|err| panic!("McFly error: Unable to commit transaction: ({err})"));

        schema::first_time_setup(&connection);

        println!("done.");

        History {
            connection,
            matcher: RefCell::new(None),
            learner: RefCell::new(None),
            save_frequency: 1,
            update_count: RefCell::new(0),
        }
    }

    fn from_db_path(path: PathBuf) -> History {
        let connection = Connection::open(path)
            .unwrap_or_else(|err| panic!("McFly error: Unable to open history database ({err})"));
        db_extensions::add_db_functions(&connection);
        History {
            connection,
            matcher: RefCell::new(None),
            learner: RefCell::new(None),
            save_frequency: 1,
            update_count: RefCell::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_is_lazy_initialized() {
        // Create an in-memory DB so we don't touch the filesystem.
        let connection = Connection::open_in_memory().unwrap();
        db_extensions::add_db_functions(&connection);

        let history = History {
            connection,
            matcher: RefCell::new(None),
            learner: RefCell::new(None),
            save_frequency: 1,
            update_count: RefCell::new(0),
        };

        // Initially the matcher should be uninitialized (None).
        assert!(history.matcher.borrow().is_none());

        // Trigger a fuzzy match. Use simple strings so the match is found.
        let indices = history.calc_match_indices("hello world", "hello", 1);

        // After calling with fuzzy>0 the matcher should be initialized.
        assert!(history.matcher.borrow().is_some());

        // The returned indices should not be empty for this match.
        assert!(!indices.is_empty());
    }
}
