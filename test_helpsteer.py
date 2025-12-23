try:
    from datasets import load_dataset
    print("datasets library found.")
    ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
    print("Dataset loaded in streaming mode.")
    count = 0
    for sample in ds:
        print(f"Sample: {sample['prompt'][:50]}...")
        count += 1
        if count >= 3: break
except Exception as e:
    print(f"Error: {e}")
