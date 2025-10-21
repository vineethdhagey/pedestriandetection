import argparse

def main():
    parser = argparse.ArgumentParser(description="Multi-view Object Detection Pipeline")
    parser.add_argument("--task", type=str, required=True, help="Task to run: preprocess, train, infer, evaluate, dashboard")
    args = parser.parse_args()

    if args.task == "preprocess":
        from src import data_preprocessing
        data_preprocessing.run()

    elif args.task == "train":
        from src import train
        train.run()

    elif args.task == "infer":
        from src import infer
        infer.run()

    elif args.task == "evaluate":
        from src import evaluate
        evaluate.run()

    elif args.task == "dashboard":
        from src import dashboard
        dashboard.run()

    else:
        print(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()
