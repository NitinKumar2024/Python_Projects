def is_prime(number):

    if number <= 1:
        return False
    elif number == 2:
        return True
    else:
        for i in range(2, int(number ** 0.5) + 1):
            if number % i == 0:
                return False
    return True

# Example usage:

while True:

    try:

        num_to_check = int(input("Enter your Number: "))
        if num_to_check == 0:
            break

        if is_prime(num_to_check):
            print(f"{num_to_check} is a prime number.")
        else:
            print(f"{num_to_check} is not a prime number.")

    except Exception as e:
        print(str(e))
