import re
import os
import xlwt
import numpy as np

data_path = 'vas-data'

feature_dir = 'feature'
if not os.path.isdir(feature_dir):
    os.makedirs(feature_dir)


def get_file_text(file_path):
    text_file = []
    duration_file = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.strip().replace('	', ' ')
            if line.startswith('*'):
                text = text.split(':', maxsplit=1)[1] + ' '
                # temp_idx = idx
                # while not '' in lines[temp_idx]:
                #     temp_idx += 1
                #     text += lines[temp_idx].strip() + ' '
                split = text.split('')
                text_part = split[0]
                if len(split) > 1:
                    duration = split[1].strip().split('_')
                    duration = int(duration[1]) - int(duration[0])
                else:
                    duration = 0
                # text = format_text(text)
                text_file.append(text_part)
                duration_file.append(duration)

    return text_file, duration_file


def format_text(text):
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub('[^0-9a-zA-Z \']+', '', text)
    text = text.strip().lower()
    return text


def get_commands(command_path):
    command_lines = []
    with open(command_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            commands = line.split('|')
            for idx in range(len(commands)):
                commands[idx] = format_text(commands[idx])
            command_lines.append(commands)
    return command_lines


def count_par(text_dict, duration_dict):
    par_dict = dict()
    par_duration_dict = dict()
    par_duration_mean_dict = dict()
    par_duration_std_dict = dict()
    for key, text_list in text_dict.items():
        par_dict[key] = len(text_list)
        par_duration_list = duration_dict[key]
        par_duration_dict[key] = sum(par_duration_list)
        par_duration_mean_dict[key] = float(np.mean(par_duration_list)) if par_duration_list else 0.
        par_duration_std_dict[key] = float(np.std(par_duration_list)) if par_duration_list else 0.
    return par_dict, par_duration_dict, par_duration_mean_dict, par_duration_std_dict


def count_successful(text_dict, duration_dict, command_lines):
    # count successful commands
    print('Count successful commands:')
    successful_dict = dict()
    successful_duration_dict = dict()
    successful_duration_mean_dict = dict()
    successful_duration_std_dict = dict()
    for key, text_list in text_dict.items():
        text_list_formatted = list(map(format_text, text_list))
        duration_list = duration_dict[key]
        success_count = 0
        success_duration_list = []
        for command_line in command_lines:
            for command in command_line:
                if command in text_list_formatted:
                    success_count += 1
                    success_idx = text_list_formatted.index(command)
                    success_duration_list.append(duration_list[success_idx])
                    break
        print(key, success_count)
        successful_dict[key] = success_count
        successful_duration_dict[key] = sum(success_duration_list)
        successful_duration_mean_dict[key] = float(np.mean(success_duration_list)) if success_duration_list else 0.
        successful_duration_std_dict[key] = float(np.std(success_duration_list)) if success_duration_list else 0.
    return successful_dict, successful_duration_dict, successful_duration_mean_dict, successful_duration_std_dict


def count_match(text_dict, duration_dict, command_lines):
    # count match commands
    print('Count match commands:')
    match_dict = dict()
    match_duration_dict = dict()
    match_duration_mean_dict = dict()
    match_duration_std_dict = dict()
    commands = sum(command_lines, [])
    for key, text_list in text_dict.items():
        text_list_formatted = list(map(format_text, text_list))
        duration_list = duration_dict[key]
        match_count = 0
        match_duration_list = []
        for idx, text in enumerate(text_list_formatted):
            if text in commands:
                match_count += 1
                match_duration_list.append(duration_list[idx])
        print(key, match_count)
        match_dict[key] = match_count
        match_duration_dict[key] = sum(match_duration_list)
        match_duration_mean_dict[key] = float(np.mean(match_duration_list)) if match_duration_list else 0.
        match_duration_std_dict[key] = float(np.std(match_duration_list)) if match_duration_list else 0.
    return match_dict, match_duration_dict, match_duration_mean_dict, match_duration_std_dict


def count_different(text_dict, duration_dict, command_lines):
    # print difference command
    print('Commands with difference:')
    different_dict = dict()
    different_duration_dict = dict()
    different_duration_mean_dict = dict()
    different_duration_std_dict = dict()
    commands = sum(command_lines, [])
    for key, text_list in text_dict.items():
        text_list_formatted = map(format_text, text_list)
        duration_list = duration_dict[key]
        print(key)
        different_count = 0
        different_duration_list = []
        for idx, text in enumerate(text_list_formatted):
            if '[*' in text_list[idx]:
                continue
            if text not in commands:
                print(text_list[idx])
                different_count += 1
                different_duration_list.append(duration_list[idx])
        print(key, different_count)
        different_dict[key] = different_count
        different_duration_dict[key] = sum(different_duration_list)
        different_duration_mean_dict[key] = float(np.mean(different_duration_list)) if different_duration_list else 0.
        different_duration_std_dict[key] = float(np.std(different_duration_list)) if different_duration_list else 0.
    return different_dict, different_duration_dict, different_duration_mean_dict, different_duration_std_dict


def count_fail(text_dict, duration_dict):
    fail_dict = dict()
    fail_duration_dict = dict()
    fail_duration_mean_dict = dict()
    fail_duration_std_dict = dict()
    for key, text_list in text_dict.items():
        duration_list = duration_dict[key]
        fail_count = 0
        fail_duration_list = []
        print(key)
        for idx, text in enumerate(text_list):
            if '[*' in text:
                fail_count += 1
                fail_duration_list.append(duration_list[idx])
                print(text)
        print(key, fail_count)
        fail_dict[key] = fail_count
        fail_duration_dict[key] = sum(fail_duration_list)
        fail_duration_mean_dict[key] = float(np.mean(fail_duration_list)) if fail_duration_list else 0.
        fail_duration_std_dict[key] = float(np.std(fail_duration_list)) if fail_duration_list else 0.
    return fail_dict, fail_duration_dict, fail_duration_mean_dict, fail_duration_std_dict


def to_label_dict(text_dict, duration_dict, command_lines, full_command_lines):
    label_name_list = ['Number of Total commands', 'Number of Accomplished commands ',
                       'Number of Matched commands',
                       'Number of Unmatched but recognized commands', 'Number of Unrecognized commands',
                       'Duration of Total commands', 'Duration of Accomplished commands ',
                       'Duration of Matched commands',
                       'Duration of Unmatched but recognized commands',
                       'Duration of Unrecognized commands',
                       'Duration mean of Total commands', 'Duration mean of Accomplished commands ',
                       'Duration mean of Matched commands',
                       'Duration mean of Unmatched but recognized commands',
                       'Duration mean of Unrecognized commands',
                       'Duration standard deviation of Total commands',
                       'Duration standard deviation of Accomplished commands ',
                       'Duration standard deviation of Matched commands',
                       'Duration standard deviation of Unmatched but recognized commands',
                       'Duration standard deviation of Unrecognized commands', ]

    par_dict, par_duration_dict, par_duration_mean_dict, par_duration_std_dict = \
        count_par(text_dict, duration_dict)
    successful_dict, successful_duration_dict, successful_duration_mean_dict, successful_duration_std_dict = \
        count_successful(text_dict, duration_dict, command_lines)
    different_dict, different_duration_dict, different_duration_mean_dict, different_duration_std_dict = \
        count_different(text_dict, duration_dict, full_command_lines)
    fail_dict, fail_duration_dict, fail_duration_mean_dict, fail_duration_std_dict = \
        count_fail(text_dict, duration_dict)
    match_dict, match_duration_dict, match_duration_mean_dict, match_duration_std_dict = \
        count_match(text_dict, duration_dict, full_command_lines)

    label_dict = dict()
    for key in par_dict.keys():
        label_dict[key] = [par_dict[key], successful_dict[key],
                           match_dict[key],
                           different_dict[key], fail_dict[key],
                           par_duration_dict[key],
                           successful_duration_dict[key], match_duration_dict[key],
                           different_duration_dict[key], fail_duration_dict[key],
                           par_duration_mean_dict[key],
                           successful_duration_mean_dict[key], match_duration_mean_dict[key],
                           different_duration_mean_dict[key], fail_duration_mean_dict[key],
                           par_duration_std_dict[key],
                           successful_duration_std_dict[key], match_duration_std_dict[key],
                           different_duration_std_dict[key], fail_duration_std_dict[key], ]
    return label_dict, label_name_list


def write_xls(label_dict, label_name_list):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet 1')

    for idx, label_name in enumerate(label_name_list):
        ws.write(0, idx + 1, label_name)

    for row_idx, (key, label_list) in enumerate(label_dict.items()):
        ws.write(row_idx + 1, 0, key)
        for col_idx, label in enumerate(label_list):
            ws.write(row_idx + 1, col_idx + 1, label)
    wb.save(os.path.join(feature_dir, 'statistics_view.xls'))


def main():
    text_dict = dict()
    duration_dict = dict()
    for text_file_name in sorted(os.listdir(data_path)):
        if text_file_name.endswith('.cha'):
            text_file_path = os.path.join(data_path, text_file_name)
            text, duration = get_file_text(text_file_path)
            text_dict[text_file_name.split('.')[0]] = text
            duration_dict[text_file_name.split('.')[0]] = duration
            print(text_file_path)
            print(text)

    command_lines = get_commands('commands.txt')
    print(command_lines)

    full_command_lines = get_commands('full_commands.txt')
    print(full_command_lines)

    label_dict, label_name_list = to_label_dict(text_dict, duration_dict, command_lines, full_command_lines)
    write_xls(label_dict, label_name_list)


if __name__ == '__main__':
    main()
