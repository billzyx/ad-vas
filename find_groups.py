import re
import os
import jiwer
import xlwt

data_path = 'vas-data'

group_label_full = ('1-8', '9-14', '15-23', '24-27', '28-35')
command_path_full = 'full_commands.txt'
group_label = ('1-8', '9-12', '13-20', '21-22', '23-30')
command_path = 'commands.txt'


def get_file_text(file_path):
    text_file = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.strip().replace('	', ' ')
            if line.startswith('*'):
                text = text.split(':', maxsplit=1)[1] + ' '
                text = text.split('')[0]
                text_file.append(text)

    return text_file


def format_text(text):
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub('[^0-9a-zA-Z \']+', '', text)
    text = text.strip().lower()
    return text


def get_commands(command_path=command_path_full, group_label_continue=group_label_full):
    command_groups = []
    with open(command_path, 'r') as f:
        lines = f.readlines()
        for group in group_label_continue:
            start = int(group.split('-')[0]) - 1
            end = int(group.split('-')[1])
            command_group = []
            for idx in range(start, end):
                commands = lines[idx].split('|')
                for commands_idx in range(len(commands)):
                    commands[commands_idx] = format_text(commands[commands_idx])
                command_group.append(commands)
            command_groups.append(command_group)
    return command_groups


def find_groups_index(text_list):
    text_list_formatted = list(map(format_text, text_list))
    groups_index_list = []
    current_idx = 0
    for current_group in range(len(command_groups_full)):
        match_in_group_idx = -1
        group_start_idx = current_idx
        while current_idx < len(text_list_formatted):
            while current_idx < len(text_list_formatted) and \
                    text_list_formatted[current_idx] in sum(command_groups_full[current_group], []):
                for alter_idx, command_group_alter_list in enumerate(command_groups_full[current_group]):
                    if text_list_formatted[current_idx] in command_group_alter_list:
                        match_in_group_idx = alter_idx
                current_idx += 1
            if not current_idx < len(text_list_formatted):
                break
            if text_list_formatted[current_idx] == '':
                current_idx += 1
                continue
            if match_in_group_idx == len(command_groups_full[current_group]) - 1:
                break
            if current_group + 1 < len(command_groups_full):
                if text_list_formatted[current_idx] in sum(command_groups_full[current_group + 1], []):
                    break
                if len(text_list_formatted[current_idx].split()) < 3:
                    current_idx += 1
                    continue
                # wer_in_group = min([jiwer.wer(text_list_formatted[current_idx], x)
                #                     for x in sum(command_groups[current_group][match_in_group_idx:], [])])
                # wer_next_group = min([jiwer.wer(text_list_formatted[current_idx], x)
                #                       for x in sum(command_groups[current_group + 1][:3], [])])
                # if wer_in_group > wer_next_group:
                #     break
            current_idx += 1
        groups_index_list.append([(group_start_idx, current_idx - 1)])
    while current_idx < len(text_list_formatted):
        for group_idx, commands in enumerate(command_groups):
            if text_list_formatted[current_idx] in sum(commands, []):
                temp_idx = current_idx + 1
                while temp_idx < len(text_list_formatted) and \
                        text_list_formatted[temp_idx] in sum(command_groups_full[group_idx], []):
                    temp_idx += 1
                if temp_idx == current_idx + 1:
                    groups_index_list[group_idx].append((current_idx,))
                else:
                    groups_index_list[group_idx].append((current_idx, temp_idx - 1))
                break
        current_idx += 1
    return groups_index_list


def find_conversation_index(text_list):
    text_list_formatted = list(map(format_text, text_list))
    conversation_idx = -1
    for idx, text in enumerate(text_list_formatted):
        if 'daughter' in text:
            conversation_idx = idx
            break
    idx = conversation_idx + 1
    while 'alexa' not in text_list_formatted[idx] and '[*' not in text_list[idx]:
        idx += 1
    if conversation_idx == idx - 1:
        return [(conversation_idx,)]
    else:
        return [(conversation_idx, idx - 1)]


def find_call_index(text_list):
    text_list_formatted = list(map(format_text, text_list))
    call_index_list = []
    idx = 0
    while idx < len(text_list_formatted):
        text = text_list_formatted[idx]
        if 'call' in text or 'six oh three' in text or 'six zero three' in text:
            temp_idx = idx + 1
            while 'alexa' not in text_list_formatted[temp_idx] and '[*' not in text_list[temp_idx]:
                temp_idx += 1
            if idx == temp_idx - 1:
                call_index_list.append((idx,))
            else:
                call_index_list.append((idx, temp_idx - 1))
        idx += 1
    return call_index_list


def write_xls(label_dict):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet 1')

    label_name_list = ['Question', 'Music', 'Reminder/Alarm/Timer/List', 'Phone Call', 'Smart Home', 'Conversation',
                       'Call']
    for idx, label_name in enumerate(label_name_list):
        ws.write(0, idx + 1, label_name)

    for row_idx, (key, label_list) in enumerate(label_dict.items()):
        ws.write(row_idx + 1, 0, key)
        for col_idx, label in enumerate(label_list):
            label_name = ''
            for label_item in label:
                assert len(label_item) <= 2
                if len(label_item) == 2:
                    label_name += str(label_item[0] + 1) + '-' + str(label_item[1] + 1) + ', '
                else:
                    label_name += str(label_item[0] + 1) + ', '
            label_name = label_name[:-2]
            ws.write(row_idx + 1, col_idx + 1, label_name)
    wb.save('group_label_auto.xls')


if __name__ == '__main__':
    text_dict = dict()
    for text_file_name in sorted(os.listdir(data_path)):
        if text_file_name.endswith('.cha'):
            text_file_path = os.path.join(data_path, text_file_name)
            text = get_file_text(text_file_path)
            text_dict[text_file_name.split('.')[0]] = text
            print(text_file_path)
            print(text)

    command_groups_full = get_commands()
    print(command_groups_full)

    command_groups = get_commands(command_path, group_label)
    print(command_groups)

    group_label_dict = dict()
    for key, text_list in text_dict.items():
        groups_index_list = find_groups_index(text_list)
        groups_index_list.append(find_conversation_index(text_list))
        groups_index_list.append(find_call_index(text_list))
        print(key, groups_index_list)
        group_label_dict[key] = groups_index_list

    write_xls(group_label_dict)
