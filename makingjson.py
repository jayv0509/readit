

def makejson(bounding_boxes,lang_list,pred):
    import json
    

    print("you have got till here")

    final_output = []
    for i in range(len(bounding_boxes)):
        item = {}
        item['box'] = [[int(coord) for coord in point] for point in bounding_boxes[i]]
        item['text'] = pred[i]
        item['language'] = lang_list[i]
        item['box'] = json.loads("[" + ", ".join(f"[{x}, {y}]" for x, y in item['box']) + "]")
        final_output.append(item)
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    return final_output


