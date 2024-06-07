# for python 3.8+
import json
from subprocess import run
import argparse
from os.path import join, exists

def chooseSrc4Attacked(src_dir, dest_root, save_dir):
    attack_result_file = join(save_dir, 'result_file.json')
    with open(attack_result_file, 'r') as f:
        data = json.load(f)

    for problem_id in data.keys():
        command = 'find '+src_dir+' -maxdepth 1 -type d -iname "*'+problem_id+'*"'
        process = run(command, shell=True, capture_output=True)
        result_dir = process.stdout.decode('utf-8').strip().split('\n')[0]

        success_results = data[problem_id]
        for record in success_results:
            author_pair = record.strip().split('from')[1].split('to')
            original = author_pair[0].strip()
            target = author_pair[1].strip()
            file_name = "_".join([problem_id, target])+".cpp"
            source_file = join(result_dir, target, original, file_name)
            new_file_name = "_".join([problem_id, target, original])+".cpp"
            dest_dir = join(dest_root, "AuthorsDirectory", original)
            dest_file = join(dest_dir, new_file_name)
            if not exists(dest_dir):
                run("mkdir -p "+dest_dir, shell=True)
            cp_cmd = 'cp '+source_file+' '+dest_file
            run(cp_cmd, shell=True)
            print(cp_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose successfully attacked source files')
    parser.add_argument('--srcdir', type=str, help='the root dir of source file')
    parser.add_argument('--destdir', type=str, help='the root dir of dest')
    parser.add_argument('--save', type=str, help='the path to save result', default='/media/yihua/2TB/backup/attack_test/code-imitator/src/PyProject')
    args = parser.parse_args()
    chooseSrc4Attacked(args.srcdir, args.destdir, args.save)