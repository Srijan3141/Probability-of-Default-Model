import argparse
from predictor import prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilty of Default Prediction")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    prediction(args.input_csv, args.output_csv)