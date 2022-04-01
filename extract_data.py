ITERATIONS = 10000


def main():
    with open('Pet_Supplies_5.json') as in_file:
        with open(f'pet_supplies_{ITERATIONS}.json', 'w') as out_file:
            for _ in range(ITERATIONS):
                line = in_file.readline()
                out_file.write(line)


if __name__ == '__main__':
    main()
