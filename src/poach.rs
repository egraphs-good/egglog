use clap::{Parser, ValueEnum};


#[derive(Clone, Copy, ValueEnum)]
enum Modes {
    Train,
    Serve,
    FineTune,
    Test,
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    mode: Modes,
}

pub fn poach (/* TODO */) {
    let args = Args::parse();
    match args.mode {
        Modes::Train => {
            train();
        }
        Modes::Serve => {
            serve();
        }
        Modes::FineTune => {
            fine_tune();
        }
        Modes::Test => {
            println!("test()");
        }
    }
    // TODO handle report IO
}

fn train() {
    println!("train()");
    //TODO
}

fn serve() {
    println!("serve()");
    //TODO
}

fn fine_tune() {
    println!("fine_tune()");
    //TODO
}
