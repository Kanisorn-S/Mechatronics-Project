import random

def randomize_numbers():
    return [random.randint(3, 5) for _ in range(3)]

if __name__ == "__main__":
    numbers = randomize_numbers()
    print(numbers)
    print(sum(numbers))
