use std::io::{self, BufWriter, Write};

use crate::cli::DumpFormat;
use crate::history::{DumpCommand, History};
use crate::settings::Settings;

#[derive(Debug)]
pub struct Dumper<'a> {
    settings: &'a Settings,
    history: &'a History,
}

impl<'a> Dumper<'a> {
    #[inline]
    pub fn new(settings: &'a Settings, history: &'a History) -> Self {
        Self { settings, history }
    }

    pub fn dump(&self) {
        let mut commands = self
            .history
            .dump(&self.settings.time_range, &self.settings.sort_order);
        if commands.is_empty() {
            println!("McFly: No history");
            return;
        }

        if let Some(pat) = &self.settings.pattern {
            commands.retain(|dc| pat.is_match(&dc.cmd));
        }

        match (self.settings.dump_format, self.settings.full_dump) {
            (DumpFormat::Json, false) => Self::dump2json(&commands),
            (DumpFormat::Csv, false) => Self::dump2csv(&commands),
            (DumpFormat::Json, true) => Self::dump2json_full(&commands),
            (DumpFormat::Csv, true) => Self::dump2csv_full(&commands),
        }
        .unwrap_or_else(|err| panic!("McFly error: Failed while output history ({err})"));
    }
}

impl Dumper<'_> {
    fn dump2json(commands: &[DumpCommand]) -> io::Result<()> {
        let mut stdout = BufWriter::new(io::stdout().lock());
        let minimal: Vec<_> = commands.iter().map(|dc| {
            serde_json::json!({
                "cmd": dc.cmd,
                "when_run": crate::time::to_datetime(dc.when_run),
            })
        }).collect();
        serde_json::to_writer_pretty(&mut stdout, &minimal).map_err(io::Error::from)?;
        stdout.flush()
    }

    fn dump2csv(commands: &[DumpCommand]) -> io::Result<()> {
        let mut wtr = csv::Writer::from_writer(io::stdout().lock());
        wtr.write_record(["cmd", "when_run"])?;
        for dc in commands {
            wtr.write_record([
                dc.cmd.as_str(),
                &crate::time::to_datetime(dc.when_run)
            ])?;
        }
        wtr.flush()
    }

    fn dump2json_full(commands: &[DumpCommand]) -> io::Result<()> {
        let mut stdout = BufWriter::new(io::stdout().lock());
        serde_json::to_writer_pretty(&mut stdout, commands).map_err(io::Error::from)?;
        stdout.flush()
    }

    fn dump2csv_full(commands: &[DumpCommand]) -> io::Result<()> {
        let mut wtr = csv::Writer::from_writer(io::stdout().lock());
        wtr.write_record([
            "id", "cmd", "cmd_tpl", "session_id", "when_run", "exit_code", "selected", "dir", "old_dir",
        ])?;
        for dc in commands {
            wtr.write_record([
                &dc.id.to_string(),
                dc.cmd.as_str(),
                dc.cmd_tpl.as_str(),
                dc.session_id.as_str(),
                &crate::time::to_datetime(dc.when_run),
                &dc.exit_code.to_string(),
                &dc.selected.to_string(),
                dc.dir.as_deref().unwrap_or(""),
                dc.old_dir.as_deref().unwrap_or(""),
            ])?;
        }
        wtr.flush()
    }
}
