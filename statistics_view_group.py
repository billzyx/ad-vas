import os
import xlwt
import xlrd

import statistics_view
import find_groups

group_label_path = 'group_label_auto_checked.xls'

feature_dir = 'feature'
if not os.path.isdir(feature_dir):
    os.makedirs(feature_dir)


def get_group_label():
    workbook = xlrd.open_workbook(group_label_path)
    worksheet = workbook.sheet_by_index(0)
    first_col = []
    for row in range(0, worksheet.nrows):
        v = worksheet.cell_value(row, 0)
        first_col.append(v)

    def get_xlsx_col_contents_single(col_num):
        elm_to_get = {}
        for row in range(1, worksheet.nrows):
            elm_to_get[first_col[row]] = set()
            v = str(worksheet.cell_value(row, col_num))
            if v != '':
                for single_range in v.split(','):
                    single_range = single_range.strip()
                    if single_range.isnumeric():
                        elm_to_get[first_col[row]].add(int(single_range) - 1)
                    else:
                        for i in range(int(single_range.split('-')[0]) - 1, int(single_range.split('-')[1])):
                            elm_to_get[first_col[row]].add(i)
            elm_to_get[first_col[row]] = list(elm_to_get[first_col[row]])
        print(elm_to_get)
        return elm_to_get

    return [get_xlsx_col_contents_single(i) for i in range(1, 8)]


def to_label_dict_group(text_dict, duration_dict, command_groups, command_groups_full):
    group_labels = get_group_label()
    print(group_labels)

    label_dict_group_list = []
    label_name_list = []

    for group_idx, single_group_label in enumerate(group_labels):
        text_dict_group = dict()
        duration_dict_group = dict()
        for key, line_num_list in single_group_label.items():
            text_dict_group[key] = [text_dict[key][idx] for idx in line_num_list]
            duration_dict_group[key] = [duration_dict[key][idx] for idx in line_num_list]
        label_dict_group, label_name_list = statistics_view.to_label_dict(
            text_dict_group, duration_dict_group, command_groups[group_idx], command_groups_full[group_idx])
        label_dict_group_list.append(label_dict_group)
    return label_dict_group_list, label_name_list


def write_xls(label_dict_group_list, label_name_list):
    wb = xlwt.Workbook()
    for idx, label_dict_group in enumerate(label_dict_group_list):
        ws = wb.add_sheet('Group ' + str(idx + 1))

        for idx, label_name in enumerate(label_name_list):
            ws.write(0, idx + 1, label_name)

        for row_idx, (key, label_list) in enumerate(label_dict_group.items()):
            ws.write(row_idx + 1, 0, key)
            for col_idx, label in enumerate(label_list):
                ws.write(row_idx + 1, col_idx + 1, label)
    wb.save(os.path.join(feature_dir, 'statistics_view_group.xls'))


def main():
    text_dict = dict()
    duration_dict = dict()
    for text_file_name in sorted(os.listdir(statistics_view.data_path)):
        if text_file_name.endswith('.cha'):
            text_file_path = os.path.join(statistics_view.data_path, text_file_name)
            text, duration = statistics_view.get_file_text(text_file_path)
            text_dict[text_file_name.split('.')[0]] = text
            duration_dict[text_file_name.split('.')[0]] = duration
            print(text_file_path)
            print(text)

    group_label_full = list(find_groups.group_label_full)
    group_label_full.extend(['18-18', '24-24'])

    group_label = list(find_groups.group_label)
    group_label.extend(['16-16', '21-21'])

    command_groups_full = find_groups.get_commands(
        command_path=find_groups.command_path_full, group_label_continue=group_label_full)
    print(command_groups_full)

    command_groups = find_groups.get_commands(
        command_path=find_groups.command_path, group_label_continue=group_label)
    print(command_groups)

    label_dict_group_list, label_name_list = \
        to_label_dict_group(text_dict, duration_dict, command_groups, command_groups_full)
    write_xls(label_dict_group_list, label_name_list)


if __name__ == '__main__':
    main()
