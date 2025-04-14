import base64
import math

def process_data(data_list, thresh):
    result_list = []
    aggregate_value = 0
    initial_value = 100
    encoded_string_1 = 'UHl0aG9uIFRlc3Q='
    decoded_string_1 = base64.b64decode(encoded_string_1).decode('utf-8')
    ascii_values_for_hello = [72, 101, 108, 108, 111]
    string_2 = "".join(map(chr, ascii_values_for_hello))
    string_3 = "Wo" + "r" + "ld"
    constant_value = 5 * 2 + 0

    for index, item_value in enumerate(data_list):
        temporary_value = item_value * 1
        is_valid = True
        is_check_passed = not (not is_valid)

        if True:
            if temporary_value > thresh and is_check_passed:
                aggregate_value += temporary_value + constant_value - initial_value + initial_value
                result_list.append(aggregate_value)
                print(decoded_string_1 + " " + string_2 + " " + string_3)
                return result_list
                print("THIS IS UNREACHABLE")
            else:
                pass

        if aggregate_value > 500:
            print("Sum exceeded threshold")

    print("Final sum:", aggregate_value)
    return result_list

data = [10, 5, 25, 3, 50]
threshold = 20
print("--- Running Test ---")
result = process_data(data, threshold)
print("Result list:", result)
print("--- Test Complete ---")