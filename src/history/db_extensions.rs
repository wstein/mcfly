use crate::history::history::Features;
use crate::ml::online::SimpleMlp;
use crate::settings::Settings;
use rusqlite::functions::FunctionFlags;
use rusqlite::Connection;

pub fn add_db_functions(db: &Connection) {
    // Try to load an on-disk SimpleMlp model; if missing, use None.
    let model = match Settings::mcfly_db_path().parent() {
        Some(p) => SimpleMlp::load(&p.join("online-ml.yaml")),
        None => None,
    };
    db.create_scalar_function(
        "nn_rank",
        10,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            let age_factor = ctx.get::<f64>(0)?;
            let length_factor = ctx.get::<f64>(1)?;
            let exit_factor = ctx.get::<f64>(2)?;
            let recent_failure_factor = ctx.get::<f64>(3)?;
            let selected_dir_factor = ctx.get::<f64>(4)?;
            let dir_factor = ctx.get::<f64>(5)?;
            let overlap_factor = ctx.get::<f64>(6)?;
            let immediate_overlap_factor = ctx.get::<f64>(7)?;
            let selected_occurrences_factor = ctx.get::<f64>(8)?;
            let occurrences_factor = ctx.get::<f64>(9)?;

            let _features = Features {
                age_factor,
                length_factor,
                exit_factor,
                recent_failure_factor,
                selected_dir_factor,
                dir_factor,
                overlap_factor,
                immediate_overlap_factor,
                selected_occurrences_factor,
                occurrences_factor,
            };

            // If model exists, score with it (model.score now accepts f64). Otherwise return 0.0.
            let score = match &model {
                Some(m) => {
                    let vec = m.score(&[
                        age_factor,
                        length_factor,
                        exit_factor,
                        recent_failure_factor,
                        selected_dir_factor,
                        dir_factor,
                        overlap_factor,
                        immediate_overlap_factor,
                        selected_occurrences_factor,
                        occurrences_factor,
                    ]);
                    vec.get(0).cloned().unwrap_or(0.0)
                }
                None => 0.0,
            };
            Ok(score)
        },
    )
    .unwrap_or_else(|err| panic!("McFly error: Successful create_scalar_function ({err})"));
}
