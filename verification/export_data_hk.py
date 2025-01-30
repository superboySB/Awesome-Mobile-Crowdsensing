import wandb

api = wandb.Api()
run = api.run("aequatio/awesome-mcs/3mz40utj")
test_frame = run.scan_history()
dataframe = run.history(samples=8400)
# export dataframe to csv
dataframe.to_csv('test-mcs.csv')
