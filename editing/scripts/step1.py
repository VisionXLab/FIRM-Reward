import json
import argparse

def extract_scores_and_average(entry) -> float:
    # --- 修改开始 ---
    # 增加类型检查：如果 entry 不是字符串，直接跳过
    if not isinstance(entry, str):
        return None
    # --- 修改结束 ---

    lines = entry.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(': ')
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None

def compute_averages(result_json_dict):
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        # 如果 avg 是 None (包括解析失败或类型不对)，则不写入结果
        if avg is not None:
            result[key] = avg
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for each key and save it as a new JSON file")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result_json json')
    parser.add_argument('--average_score_json', type=str, required=True, help='Path of average_score_json json')

    args = parser.parse_args()

    try:
        with open(args.result_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        averaged_data = compute_averages(data)

        with open(args.average_score_json, 'w', encoding='utf-8') as f:
            json.dump(averaged_data, f, indent=2)
            
        print(f"Successfully processed. Saved to {args.average_score_json}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()