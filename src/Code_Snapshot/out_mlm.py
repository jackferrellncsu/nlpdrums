import pickle
import os

result_objs = []

for i in range(5):
    with open(f"Snapshot_Outs/OutFiles_mlm/results_0_{i}.pkl", "rb") as file:
        
        result = pickle.load(file)
    result_objs.append(result)

lengthst_95 = []
lengthst_90 = []
lengthst_80 = []
lengthst_75 = []

means_95 = []
means_90 = []
means_80 = []
means_75 = []

emp_confs_95 = []
emp_confs_90 = []
emp_confs_80 = []
emp_confs_75 = []

cas = []
ops = []
ofs = []
creds = []


for i in range(len(result_objs)):
    lengthst_95 = lengthst_95 + result_objs[i]['lengths_95']
    lengthst_90 = lengthst_90 + result_objs[i]['lengths_90']
    lengthst_80 = lengthst_80 + result_objs[i]['lengths_80']
    lengthst_75 = lengthst_75 + result_objs[i]['lengths_75']

    means_95.append(result_objs[i]['mean_95'])
    means_90.append(result_objs[i]['mean_90'])
    means_80.append(result_objs[i]['mean_80'])
    means_75.append(result_objs[i]['mean_75'])

    emp_confs_95.append(result_objs[i]['emp_conf_95'])
    emp_confs_90.append(result_objs[i]['emp_conf_90'])
    emp_confs_80.append(result_objs[i]['emp_conf_80'])
    emp_confs_75.append(result_objs[i]['emp_conf_75'])

    cas.append(result_objs[i]['classification_accuracy'])
    ops.append(result_objs[i]['OP'])
    ofs.append(result_objs[i]['OF'])
    creds.append(result_objs[i]['credibility'])

#Reportable results:
#Median:
med_tot_95 = statistics.median(lengthst_95)
med_tot_90 = statistics.median(lengthst_90)
med_tot_80 = statistics.median(lengthst_80)
med_tot_75 = statistics.median(lengthst_75)

print(f'Median Set Size 95%: {med_tot_95}')
print(f'Median Set Size 90%: {med_tot_90}')
print(f'Median Set Size 80%: {med_tot_80}')
print(f'Median Set Size 75%: {med_tot_75}')

#Mean:
mean_tot_95 = statistics.mean(means_95)
mean_tot_90 = statistics.mean(means_90)
mean_tot_80 = statistics.mean(means_80)
mean_tot_75 = statistics.mean(means_75)

print(f'Mean Set Size 95%: {mean_tot_95}')
print(f'Mean Set Size 90%: {mean_tot_90}')
print(f'Mean Set Size 80%: {mean_tot_80}')
print(f'Mean Set Size 75%: {mean_tot_75}')

#Confidences:
conf_tot_95 = statistics.mean(emp_confs_95)
conf_tot_90 = statistics.mean(emp_confs_90)
conf_tot_80 = statistics.mean(emp_confs_80)
conf_tot_75 = statistics.mean(emp_confs_75)

print(f'Average Empirical Confidence 95%: {conf_tot_95}')
print(f'Average Empirical Confidence 90%: {conf_tot_90}')
print(f'Average Empirical Confidence 80%: {conf_tot_80}')
print(f'Average Empirical Confidence 75%: {conf_tot_75}')

#OPs:
op_tot = statistics.mean(ops)
print(f'Average OP Score: {op_tot}')

#OFs:
of_tot = statistics.mean(ofs)
print(f'Average OF Score: {of_tot}')

#CA:
ca_tot = statistics.mean(cas)
print(f'Average Classification Accuracy: {ca_tot}')

#Cred:
cred_tot = statistics.mean(creds)
print(f'Average Credibility: {cred_tot}')