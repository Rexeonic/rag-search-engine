import argparse

from lib.evaluation import ManualEvaluation
from lib.preprocessing import GetData

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    # run evaluation logic here
    golden_dataset = GetData('golden_dataset.json').get_file_data_json()
    # test_cases -> list[{query: "", relevant_docs: [""] }]
    test_cases = golden_dataset['test_cases']

    ManualEvaluation().evaluate(test_cases, limit)

if __name__ == "__main__":
    main()