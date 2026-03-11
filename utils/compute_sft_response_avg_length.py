import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sft" / "sft_train.jsonl"



def get_response_text(record: dict) -> str | None:
    """Return the last assistant response text from one SFT record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def compute_average_response_length(input_path: Path) -> tuple[int, float]:
    """Compute average assistant response length in characters."""
    total_length = 0
    response_count = 0

    with input_path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

            response_text = get_response_text(record)
            if response_text is None:
                continue

            total_length += len(response_text)
            response_count += 1

    if response_count == 0:
        raise ValueError("No assistant responses were found in the input file.")

    average_length = total_length / response_count
    return response_count, average_length


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the average length of assistant responses in an SFT JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the SFT JSONL file (default: {DEFAULT_INPUT_PATH})",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    response_count, average_length = compute_average_response_length(input_path)
    print(f"file: {input_path}")
    print(f"responses: {response_count}")
    print(f"average_response_length_chars: {average_length:.2f}")


if __name__ == "__main__":
    main()
