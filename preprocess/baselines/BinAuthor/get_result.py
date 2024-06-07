import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--file", type=str, help="path to your idapy_output.txt", default="X:\\idapy_output.txt")
args = parser.parse_args()

file = args.file
f = open(file, 'r')
total_num = 0.0
correct = 0.0
total_time = 0
for count, line in enumerate(f, start=1):
    # print(count, line[:20])
    if count % 4 == 3:
        continue
    elif count % 4 == 2:
        line = line.strip('\n')
        cur_time = line.split(' ')[-1]
        total_time += float(cur_time)
    elif count % 4 == 1:
        line = line.strip('\n')
        ground_truth = line.split(' ')[-1].strip()
        # print(ground_truth)
    else:
        line = line.strip('\t ')
        predict_value = line.split(' ')[0]
        total_num += 1
        if ground_truth == predict_value:
            correct += 1
acc = 100* correct / total_num
print(" there are {} test records, correct : {}\t accuracy rate : {:4.2f}%".format(total_num, correct, acc))
print(" totol time-consumming : {}\t average time-consuming : {:6.2f}".format(total_time, total_time/total_num))