import numpy as np


def main(sigma, nums=50000):
    max_a = 0
    temp = []
    count = 0
    for i in range(1, nums):
        a = np.random.normal(0, sigma * 2)
        temp.append(abs(a))
        if abs(a) > sigma * 2:
            count += 1
        if abs(a) > max_a:
            max_a = abs(a)

    print(count)
    print(max_a)
    print(np.mean(temp))


if __name__ == '__main__':
    sigma_angle = 0.2
    sigma_offset = 12.0

    main(sigma_angle)
    main(sigma_offset)

