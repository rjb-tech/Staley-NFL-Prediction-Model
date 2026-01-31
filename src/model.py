import os
import uuid
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta


@dataclass
class TrainingMetrics:
    pass


@dataclass
class TrainingParams:
    train_start: datetime | None = None
    train_end: datetime | None = None
    train_duration: timedelta | None = None
    n_estimators: int | None = None
    max_depth: int | None = None
    learning_rate: float | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None


class Staley:
    def __init__(self, args):
        self.training_args = args
        self.training_metrics = TrainingMetrics()

        self.run_hash: str = uuid.uuid4().hex[:10]
        date_string = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.model_folder: str = os.path.join(
            os.getcwd(), "models", f"{date_string}_{self.run_hash}"
        )

        self.training_params = TrainingParams(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )

    def __start_training(self):
        if self.training_params.train_start is not None:
            raise ValueError("Training has already been started.")
        elif self.training_params.train_end is not None:
            raise ValueError("Training has already been completed.")
        self.training_params.train_start = datetime.now()

    def __end_training(self):
        if not self.training_params.train_start:
            raise ValueError("Training has not been started.")
        elif self.training_params.train_end is not None:
            raise ValueError("Training has already been completed.")
        self.training_params.train_end = datetime.now()
        self.training_params.train_duration = (
            self.training_params.train_end - self.training_params.train_start
        )

        os.makedirs(self.model_folder, exist_ok=False)
        self.__save_log_file()

    def __save_log_file(self):
        log_path = os.path.join(self.model_folder, f"{self.run_hash}-training.log")
        with open(log_path, "w") as f:
            f.write("=== Training Params ===\n")
            for key, value in asdict(self.training_params).items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("\n=== Training Metrics ===\n")
            for key, value in asdict(self.training_metrics).items():
                f.write(f"{key}: {value}\n")
        print(f"Training log saved to {log_path}")

    def train(self):
        self.__start_training()

        # training logic goes here

        self.__end_training()


def train_command(args):
    model = Staley(args)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Staley NFL Prediction Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of boosting rounds."
    )
    train_parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum tree depth."
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Boosting learning rate."
    )
    train_parser.add_argument(
        "--subsample", type=float, default=0.8, help="Fraction of training data per tree."
    )
    train_parser.add_argument(
        "--colsample-bytree", type=float, default=0.8, help="Fraction of features per tree."
    )
    train_parser.add_argument(
        "--reg-alpha", type=float, default=0.0, help="L1 regularization."
    )
    train_parser.add_argument(
        "--reg-lambda", type=float, default=1.0, help="L2 regularization."
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)


if __name__ == "__main__":
    main()
