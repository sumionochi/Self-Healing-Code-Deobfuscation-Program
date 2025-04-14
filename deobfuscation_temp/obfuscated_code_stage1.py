import base64
import math

def filter_and_sum_values(input_values, thresh):
    filtered_results = []
    aggregated_value = 0
    initial_value = 100
    encoded_string_1 = 'UHl0aG9uIFRlc3Q='
    decoded_greeting_message = base64.b64decode(encoded_string_1).decode('utf-8')
    string_2_character_codes = [72, 101, 108, 108, 111]
    greeting_message = "".join(map(chr, string_2_character_codes))
    greeting_suffix = "Wo" + "r" + "ld"
    constant_value = 5 * 2 + 0

    for index, list_element in enumerate(input_values):
        scaled_data_value = list_element * 1
        is_data_valid = True
        is_check_passed = not (not is_data_valid)

        if True:
            if scaled_data_value > thresh and is_check_passed:
                aggregated_value += scaled_data_value + constant_value - initial_value + initial_value
                filtered_results.append(aggregated_value)
                print(decoded_greeting_message + " " + greeting_message + " " + greeting_suffix)
                return filtered_results
                print("THIS IS UNREACHABLE")
            else:
                pass

        if aggregated_value > 500:
            print("Sum exceeded threshold")

    print("Final sum:", aggregated_value)
    return filtered_results

data = [10, 5, 25, 3, 50]
threshold = 20
print("--- Running Test ---")
result = filter_and_sum_values(data, threshold)
print("Result list:", result)
print("--- Test Complete ---")