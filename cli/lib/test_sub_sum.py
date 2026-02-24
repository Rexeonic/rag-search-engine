from semantic_search import add_vectors, subtract_vectors

TestCase = tuple[list[float], list[float], str, list[float] | str]

run_cases: list[TestCase] = [
    ([1, 2, 3], [4, 5, 6], "add", [5, 7, 9]),
    ([4, 5, 6], [1, 2, 3], "subtract", [3, 3, 3]),
    ([1, -2, 3], [-4, 5, -6], "add", [-3, 3, -3]),
    ([1, -2, 3], [-4, 5, -6], "subtract", [5, -7, 9]),
]

submit_cases: list[TestCase] = run_cases + [
    ([0.5, 0.25], [0.25, 0.75], "add", [0.75, 1.0]),
    ([1], [1, 2], "subtract", "exception"),
    ([], [], "subtract", []),
    ([1, 0, -1, 2, -2], [2, -2, 1, -1, 0], "add", [3, -2, 0, 1, -2]),
    ([1, 0, -1, 2, -2], [2, -2, 1, -1, 0], "subtract", [-1, 2, -2, 3, -2]),
]


def test(
    vec1: list[float], vec2: list[float], op: str, expected: list[float] | str
) -> bool:
    print("---------------------------------")
    print(f"Vector 1:  {vec1}")
    print(f"Vector 2:  {vec2}")
    print(f"Operation: {op}")
    print(f"\nExpected result: {expected}")

    try:
        result = (
            add_vectors(vec1, vec2) if op == "add" else subtract_vectors(vec1, vec2)
        )
        print(f"Actual result:   {result}")
    except ValueError:
        print("Actual result:   exception")
        if expected == "exception":
            print("\nPass")
            return True
        else:
            print("\nFail")
            return False
    except Exception as e:
        print(f"Actual result:   {e}")
        print("\nFail")
        return False

    if isinstance(expected, list) and result == expected:
        print("\nPass")
        return True

    print("\nFail")
    return False


def main() -> None:
    passed = 0
    failed = 0
    skipped = len(submit_cases) - len(test_cases)

    for test_case in test_cases:
        correct = test(*test_case)
        if correct:
            passed += 1
        else:
            failed += 1

    if failed == 0:
        print("============= PASS ==============")
    else:
        print("============= FAIL ==============")

    if skipped > 0:
        print(f"{passed} passed, {failed} failed, {skipped} skipped")
    else:
        print(f"{passed} passed, {failed} failed")


test_cases: list[TestCase] = submit_cases
if "__RUN__" in globals():
    test_cases = run_cases

main()