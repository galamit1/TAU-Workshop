from pandas import read_csv, DataFrame

d = read_csv('../input/train_labeled.csv')
shuffled_df = d.sample(frac=1, random_state=42)

NUMBER_OF_EACH_CLASS = 2500
TIMES = 1

count_by_class = {0: 0, 1: 0}
output = []
for item in shuffled_df.values:
    try:
        if count_by_class[item[1]] < NUMBER_OF_EACH_CLASS:
            item[0] = str(item[0]).replace('\r\n', '. ').replace('\n', '. ')
            output.append(item)
            count_by_class[item[1]] += 1
        if count_by_class[0] == count_by_class[1] == NUMBER_OF_EACH_CLASS:
            count_by_class = {0: 0, 1: 0}
            if len(output) == NUMBER_OF_EACH_CLASS * TIMES * 2:
                break
            continue
    except:
        continue

print(count_by_class)
df = DataFrame(output)
out = df.to_csv(index=False, header=True)

output_file = 'train_labeled_{}_balanced.csv'.format(str(NUMBER_OF_EACH_CLASS * TIMES * 2))
with open(output_file, 'w') as f:
    f.write(out)