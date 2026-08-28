import os
import pandas as pd
import json
import re
import sqlite3
import sys
import Levenshtein


def _data_root():
    """Root of the released EHRSQL-EHRAgent tree — set env EHRAGENT_DATA_ROOT."""
    configured = os.environ.get("EHRAGENT_DATA_ROOT", "").strip()
    if configured:
        return configured.rstrip("/")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ehrsql-ehragent"))


def _mimic_iii_csv_dir():
    """
    Paper release (Google Drive) unpacks to ``ehrsql-ehragent/mimic_iii/*.csv``.
    Older layouts used ``ehrsql/mimic_iii/*.csv`` under the same root.
    """
    root = _data_root()
    for sub in (os.path.join(root, "ehrsql", "mimic_iii"), os.path.join(root, "mimic_iii")):
        if os.path.isfile(os.path.join(sub, "ADMISSIONS.csv")):
            return sub
    return os.path.join(root, "mimic_iii")


def db_loader(target_ehr):
    mimic = _mimic_iii_csv_dir()
    root = _data_root()
    ehr_dict = {"admissions": f"{mimic}/ADMISSIONS.csv",
                "chartevents": f"{mimic}/CHARTEVENTS.csv",
                "cost": f"{mimic}/COST.csv",
                "d_icd_diagnoses": f"{mimic}/D_ICD_DIAGNOSES.csv",
                "d_icd_procedures": f"{mimic}/D_ICD_PROCEDURES.csv",
                "d_items": f"{mimic}/D_ITEMS.csv",
                "d_labitems": f"{mimic}/D_LABITEMS.csv",
                "diagnoses_icd": f"{mimic}/DIAGNOSES_ICD.csv",
                "icustays": f"{mimic}/ICUSTAYS.csv",
                "inputevents_cv": f"{mimic}/INPUTEVENTS_CV.csv",
                "labevents": f"{mimic}/LABEVENTS.csv",
                "microbiologyevents": f"{mimic}/MICROBIOLOGYEVENTS.csv",
                "outputevents": f"{mimic}/OUTPUTEVENTS.csv",
                "patients": f"{mimic}/PATIENTS.csv",
                "prescriptions": f"{mimic}/PRESCRIPTIONS.csv",
                "procedures_icd": f"{mimic}/PROCEDURES_ICD.csv",
                "transfers": f"{mimic}/TRANSFERS.csv",
                }
    if target_ehr not in ehr_dict:
        raise KeyError(
            f"Unknown table {target_ehr!r}. Use one of: {', '.join(sorted(ehr_dict.keys()))}."
        )
    csv_path = ehr_dict[target_ehr]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. "
            f"Set env EHRAGENT_DATA_ROOT to your EHRSQL-EHRAgent root (folder that contains mimic_iii/ with ADMISSIONS.csv). "
            f"Currently EHRAGENT_DATA_ROOT={root!r}, resolved mimic_iii dir={mimic!r}."
        )
    data = pd.read_csv(csv_path)
    # data = data.astype(str)
    column_names = ', '.join(data.columns.tolist())
    return data
# def get_column_names(self, target_db):
#     return ', '.join(data.columns.tolist())

def data_filter(data, argument):
    # commands = re.sub(r' ', '', argument)
    backup_data = data
    # print('-->', argument)
    commands = argument.split('||')
    for i in range(len(commands)):
        try:
            # commands[i] = commands[i].replace(' ', '')
            if '>=' in commands[i]:
                command = commands[i].split('>=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] >= value]
            elif '<=' in commands[i]:
                command = commands[i].split('<=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] <= value]
            elif '>' in commands[i]:
                command = commands[i].split('>')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] > value]
            elif '<' in commands[i]:
                command = commands[i].split('<')
                column_name = command[0]
                value = command[1]
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] < value]
            elif '=' in commands[i]:
                command = commands[i].split('=')
                column_name = command[0]
                value = command[1]
                # print(command)
                # print(value)
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    examplar = backup_data[column_name].tolist()[0]
                    value = type(examplar)(value)
                    # print(value, type(value), type(examplar))
                except:
                    value = value
                    # print('--', value, type(value), type(examplar))
                # print('------', len(data))
                data = data[data[column_name] == value]
                # print('======', len(data))
            elif ' in ' in commands[i]:
                command = commands[i].split(' in ')
                column_name = command[0]
                value = command[1]
                value_list = [s.strip() for s in value.strip("[]").split(',')]
                value_list = [s.strip("'").strip('"') for s in value_list]
                # print(command)
                # print(column_name)
                # print(value)
                # print(value_list)
                value_list = list(map(type(data[column_name][0]), value_list))
                # print(len(data))
                data = data[data[column_name].isin(value_list)]
                # print(len(data))
            elif 'max' in commands[i]:
                command = commands[i].split('max(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].max()]
            elif 'min' in commands[i]:
                command = commands[i].split('min(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].min()]
        except:
            if column_name not in data.columns.tolist():
                columns = ', '.join(data.columns.tolist())
                raise Exception("The filtering query {} is incorrect. Please modify the column name or use LoadDB to read another table. The column names in the current DB are {}.".format(commands[i], columns))
            if column_name == '' or value == '':
                raise Exception("The filtering query {} is incorrect. There is syntax error in the command. Please modify the condition or use LoadDB to read another table.".format(commands[i]))
        if len(data) == 0:
            # get 5 examples from the backup data what is in the same column
            column_values = list(set(backup_data[column_name].tolist()))
            if ('=' in commands[i]) and (not value in column_values) and (not '>=' in commands[i]) and (not '<=' in commands[i]):
                levenshtein_dist = {}
                for cv in column_values:
                    levenshtein_dist[cv] = Levenshtein.distance(str(cv), str(value))
                levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
                column_values = [i[0] for i in levenshtein_dist[:5]]
                column_values = ', '.join([str(i) for i in column_values])
                raise Exception("The filtering query {} is incorrect. There is no {} value in the column. Five example values in the column are {}. Please check if you get the correct {} value.".format(commands[i], value, column_values, column_name))
            else:
                return data
    return data

def get_value(data, argument):
    try:
        commands = argument.split(', ')
        if len(commands) == 1:
            column = argument
            while column[0] == '[' or column[0] == "'":
                column = column[1:]
            while column[-1] == ']' or column[-1] == "'":
                column = column[:-1]
            if len(data) == 1:
                return str(data.iloc[0][column])
            else:
                answer_list = list(set(data[column].tolist()))
                answer_list = [str(i) for i in answer_list]
                return ', '.join(answer_list)
                # else:
                #     return "Get the value. But there are too many returned values. Please double-check the code and make necessary changes."
        else:
            column = commands[0]
            if 'mean' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)/len(res_list)
            elif 'max' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return max(res_list)
            elif 'min' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return min(res_list)
            elif 'sum' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)
            elif 'list' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [str(i) for i in res_list]
                return list(res_list)
            else:
                raise Exception("The operation {} contains syntax errors. Please check the arguments.".format(commands[-1]))
    except:
        column_values = ', '.join(data.columns.tolist())
        raise Exception("The column name {} is incorrect. Please check the column name and make necessary changes. The columns in this table include {}.".format(column, column_values))

def _mimic_iii_db_path():
    """SQLite DB shipped with EHRSQL-EHRAgent (optional; CSV path used by LoadDB)."""
    mimic = _mimic_iii_csv_dir()
    for name in ("mimic_iii.db", "MIMIC_III.DB"):
        path = os.path.join(mimic, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"mimic_iii.db not found under {mimic!r}. "
        f"Set EHRAGENT_DATA_ROOT (currently {_data_root()!r}) to the ehrsql-ehragent root."
    )


def sql_interpreter(command):
    db_path = _mimic_iii_db_path()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        results = cur.execute(command).fetchall()
    finally:
        con.close()
    return results

def _normalize_calendar_modifier(argument: str) -> str:
    """Map natural phrases to SQLite datetime() modifiers (see https://sqlite.org/lang_datefunc.html)."""
    a = str(argument).strip()
    if not a:
        raise ValueError("Calendar argument must be non-empty.")
    # Bare calendar year (e.g. "2104" from "since 2104") is not a SQLite modifier.
    if re.fullmatch(r"\d{4}", a):
        raise ValueError(
            f"Calendar({a!r}) is a year, not a relative duration. "
            "Use SQLInterpreter with strftime('%Y', CHARTTIME) >= '{year}' for year filters, "
            "or Calendar('-1 year') for relative offsets."
        )
    low = a.lower()
    # Phrases like "1 year ago" are not valid in SQLite; use "-1 year".
    if low.endswith(" ago"):
        body = a[:-4].strip()
        if body and not body[0] in "+-":
            body = "-" + body
        return body
    # "0 year" from old prompts = "now"
    if low in ("0 year", "0 years", "now", "today"):
        return "+0 seconds"
    return a


def date_calculator(argument):
    """
    ``Calendar(DURATION)`` → ``datetime('now', DURATION)`` in SQLite (in-memory DB; no mimic_iii.db needed).
    Valid examples: ``'-1 year'``, ``'-365 days'``, ``'+0 seconds'`` (now), ``'start of day'``.
    """
    mod = _normalize_calendar_modifier(argument)
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    try:
        row = cur.execute("select datetime('now', ?)", (mod,)).fetchone()
        if not row or row[0] is None:
            raise ValueError(
                "Invalid Calendar modifier {!r} (normalized {!r}): SQLite returned NULL. "
                "Use relative forms like '-1 year' or filter years in SQL with "
                "strftime('%Y', CHARTTIME) >= '2104'. Do not use Calendar('01/01/2104').".format(
                    argument, mod
                )
            )
        return row[0]
    except sqlite3.Error as e:
        # Bare ``365 days`` needs a sign in SQLite.
        if mod and mod[0] not in "+-" and any(
            u in mod.lower() for u in ("year", "month", "day", "hour", "minute", "second")
        ):
            try:
                (result,) = cur.execute("select datetime('now', ?)", (f"+{mod}",)).fetchone()
                return result
            except sqlite3.Error:
                pass
        raise Exception(
            "Invalid Calendar modifier {!r} (normalized {!r}): {}. "
            "Use SQLite forms like '-1 year', '-365 days', or '+0 seconds' for 'now'.".format(
                argument, mod, e
            )
        ) from e

if __name__ == "__main__":
    db = table_toolkits()
    print(db.db_loader("microbiologyevents"))
    # print(db.data_filter("SPEC_TYPE_DESC=peripheral blood lymphocytes"))
    print(db.data_filter("HADM_ID=107655"))
    print(db.data_filter("SPEC_TYPE_DESC=peripheral blood lymphocytes"))
    print(db.get_value('CHARTTIME'))
    # results = db.sql_interpreter("select max(t1.c1) from ( select sum(cost.cost) as c1 from cost where cost.hadm_id in ( select diagnoses_icd.hadm_id from diagnoses_icd where diagnoses_icd.icd9_code = ( select d_icd_diagnoses.icd9_code from d_icd_diagnoses where d_icd_diagnoses.short_title = 'comp-oth vasc dev/graft' ) ) and datetime(cost.chargetime) >= datetime(current_time,'-1 year') group by cost.hadm_id ) as t1")
    # results = [result[0] for result in results]
    # if len(results) == 1:
    #     print(results[0])
    # else:
    #     print(results)
    # print(db.date_calculator('-1 year'))
