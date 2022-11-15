


class Dataset:

    def __init__(self, mode, tokenizer, texts, summaries):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries


        self.max_length = 512
        self.max_summary_length = 64

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        summary = self.summaries[item]
        
        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding='max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.max_length]
        
        summary_inputs = self.tokenizer(summary, return_tensors="pt", max_length=self.max_summary_length, padding='max_length')
        summary_inputs["input_ids"] = summary_inputs["input_ids"][:, :self.max_summary_length]
        summary_inputs["attention_mask"] = summary_inputs["attention_mask"][:, :self.max_summary_length]

        batch = {
            "text": text,
            "text_inputs": text_inputs,
            "summary": summary,
            "summary_inputs": summary_inputs,
        }

        return batch

