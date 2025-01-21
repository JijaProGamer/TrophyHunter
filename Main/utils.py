import difflib

def filter_ocr_results(ocr_results, correct_texts, max_distance=2):
    filtered_results = []
    for result in ocr_results:
        detected_name = result[1].upper()
        bbox = result[0]

        closest_match = difflib.get_close_matches(detected_name, correct_texts, n=1, cutoff=0.0)
        if closest_match:
            match = closest_match[0]
            
            distance = levenshtein_distance(detected_name, match)

            #if match == "LEON":
            #    print(match, closest_match, detected_name)
            
            if distance <= max_distance:
                #avg_x = (bbox[0][0] + bbox[2][0]) / 2
                #avg_y = (bbox[0][1] + bbox[2][1]) / 2
                
                filtered_results.append({
                    "text": match,
                    "pos": [bbox[0][0], bbox[0][1]]
                })

    return filtered_results

def levenshtein_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]
