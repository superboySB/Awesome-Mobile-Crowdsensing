from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Get the course name and grade level
course_name = "Math"
grade_level = "3"
num_questions = 10
topics = "addition, subtraction, multiplication, and division"
prompt = f"Write an assignment for the course {course_name} for grade level {grade_level}. The assignment should be {num_questions} questions long and should cover the following topics: {topics}."

# tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# generate the assignments
assignments = model.generate(
    input_ids,
    attention_mask=input_ids,
    pad_token_id=tokenizer.eos_token_id,
    max_length=512,
    num_return_sequences=1,
    temperature=0.7,
    top_k=5,
)

# decode and Print the assignment
for assignment in assignments:
    assignment = tokenizer.decode(assignment, skip_special_tokens=True)
    print(assignment, end="\n")
