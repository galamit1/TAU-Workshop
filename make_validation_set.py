from pandas import read_csv, DataFrame

d = read_csv('test_labeled.csv')
shuffled_df = d.sample(frac=1, random_state=42)

NUMBER_OF_EACH_CLASS = 5
TIMES = 100

count_by_class = {0: 0, 1: 0}
output = []
for item in shuffled_df.values:
    if count_by_class[item[1]] < NUMBER_OF_EACH_CLASS:
        item[0] = str(item[0]).replace('\r\n', '. ').replace('\n', '. ')
        output.append(item)
        count_by_class[item[1]] += 1
    if count_by_class[0] == count_by_class[1] == NUMBER_OF_EACH_CLASS:
        count_by_class = {0: 0, 1: 0}
        if len(output) == NUMBER_OF_EACH_CLASS * TIMES:
            break
        continue

df = DataFrame(output)
out = df.to_csv(index=False, header=True)

output_file = 'test_labeled_{}_balanced.csv'.format(str(NUMBER_OF_EACH_CLASS * TIMES * 2))
with open(output_file, 'w') as f:
    f.write(out)