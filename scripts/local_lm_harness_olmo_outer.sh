#!/bin/bash

model_name="olmo1b"
task="blimp"

job_subset_name=$"lmeval-$task-$model_name"
job_subset_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_subset_name")
job_subset_name=$(echo "$job_subset_name" | tr '[:upper:]' '[:lower:]')

################################################################################
job_name="$job_subset_name-1"
echo $job_name

revision_list=$"step0-tokens0B step1000-tokens2B step2000-tokens4B step3000-tokens6B step4000-tokens8B step4500-tokens9B step5000-tokens10B step6000-tokens12B step7000-tokens14B step8000-tokens16B step9000-tokens18B step10000-tokens20B step11000-tokens23B step12000-tokens25B step13000-tokens27B step14000-tokens29B step15000-tokens31B step16000-tokens33B step17000-tokens35B step18000-tokens37B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list

################################################################################
job_name="$job_subset_name-2"
echo $job_name

revision_list=$"step19000-tokens39B step20000-tokens41B step21000-tokens44B step22000-tokens46B step23000-tokens48B step24000-tokens50B step25000-tokens52B step26000-tokens54B step27000-tokens56B step28000-tokens58B step29000-tokens60B step30000-tokens62B step31000-tokens64B step32000-tokens67B step33000-tokens69B step34000-tokens71B step35000-tokens73B step36000-tokens75B step37000-tokens77B step38000-tokens79B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list


################################################################################
job_name="$job_subset_name-3"
echo $job_name

revision_list=$"step39000-tokens81B step40000-tokens83B step41000-tokens85B step42000-tokens88B step43000-tokens90B step44000-tokens92B step45000-tokens94B step46000-tokens96B step47000-tokens98B step48000-tokens100B step49000-tokens102B step50000-tokens104B step51000-tokens106B step52000-tokens109B step53000-tokens111B step54000-tokens113B step55000-tokens115B step56000-tokens117B step57000-tokens119B step58000-tokens121B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list

################################################################################
job_name="$job_subset_name-4"
echo $job_name

revision_list=$"step59000-tokens123B step60000-tokens125B step61000-tokens127B step62000-tokens129B step63000-tokens132B step64000-tokens134B step65000-tokens136B step66000-tokens138B step67000-tokens140B step68000-tokens142B step69000-tokens144B step70000-tokens146B step71000-tokens148B step72000-tokens150B step73000-tokens153B step74000-tokens155B step75000-tokens157B step76000-tokens159B step77000-tokens161B step78000-tokens163B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list

################################################################################
job_name="$job_subset_name-5"
echo $job_name

revision_list=$"step79000-tokens165B step80000-tokens167B step81000-tokens169B step82000-tokens171B step83000-tokens174B step84000-tokens176B step85000-tokens178B step86000-tokens180B step87000-tokens182B step88000-tokens184B step89000-tokens186B step90000-tokens188B step91000-tokens190B step94000-tokens197B step95000-tokens199B step96000-tokens201B step97000-tokens203B step98000-tokens205B step99000-tokens207B step100000-tokens209B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list


################################################################################
job_name="$job_subset_name-6"
echo $job_name

revision_list=$"step101000-tokens211B step102000-tokens213B step103000-tokens215B step104000-tokens218B step105000-tokens220B step106000-tokens222B step107000-tokens224B step108000-tokens226B step109000-tokens228B step110000-tokens230B step111000-tokens232B step112000-tokens234B step113000-tokens236B step114000-tokens238B step115000-tokens241B step116000-tokens243B step117000-tokens245B step118000-tokens247B step119000-tokens249B step120000-tokens251B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list


################################################################################
job_name="$job_subset_name-7"
echo $job_name

revision_list=$"step121000-tokens253B step122000-tokens255B step123000-tokens257B step124000-tokens259B step125000-tokens262B step126000-tokens264B step127000-tokens266B step128000-tokens268B step129000-tokens270B step130000-tokens272B step131000-tokens274B step132000-tokens276B step133000-tokens278B step134000-tokens280B step135000-tokens283B step136000-tokens285B step137000-tokens287B step138000-tokens289B step139000-tokens291B step140000-tokens293B"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list

################################################################################
job_name="$job_subset_name-8"
echo $job_name

revision_list=$"step141000-tokens295B step142000-tokens297B step143000-tokens299B step1454000-tokens3048B main"
bash scripts/rcp_lm_harness_olmo_inner.sh $revision_list

################################################################################
cnt=9
echo $job_name

revision_list=$"step1143000-tokens2396B step643000-tokens1348B step743000-tokens1557B step24300-tokens509B step1443000-tokens3025B step1243000-tokens2605B step543000-tokens1138B step443000-tokens928B step343000-tokens719B step943000-tokens1976B step1343000-tokens2815B step1043000-tokens2186B step843000-tokens1767B"
for rev in $revision_list; do
    job_name="$job_subset_name-$cnt"
    echo $job_name
    
    bash scripts/rcp_lm_harness_olmo_inner.sh $rev
    ((cnt++))
done