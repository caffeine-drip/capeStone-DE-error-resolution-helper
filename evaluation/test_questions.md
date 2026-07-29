# Manual Test Questions — DE IncidentIQ

50 questions to paste into the Streamlit app (or `python rag.py "..."`) to manually explore
retrieval quality, answer quality, and the self-check/repair step. Grouped by the corpus
categories so you can check whether the right kind of document gets retrieved.

---

## Spark config tuning (full spark-submit commands)

1. My Spark job keeps failing with `java.lang.OutOfMemoryError: Java heap space` during the shuffle stage of a large groupBy aggregation. Here's my current spark-submit command:
```
spark-submit \
  --class com.mycompany.etl.DailyAggregationJob \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 4g \
  --executor-cores 4 \
  --num-executors 10 \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  s3://my-bucket/jobs/daily-aggregation.jar
```
The job processes ~500GB of data with a wide join followed by a groupBy aggregation. It ran fine on smaller datasets but started failing consistently once we scaled up to this volume. Executors are dying with OOM around the shuffle-write stage. How should I fix the memory and partition configuration?

2. My driver keeps crashing with `OutOfMemoryError` after calling `.collect()` on a large DataFrame. Current command: `spark-submit --driver-memory 4g --executor-memory 8g --executor-cores 4 --num-executors 20 my_job.jar`. The result set is larger than expected. What should I change?

3. I'm seeing heavy data skew — one task in my shuffle stage takes 45 minutes while the other 199 finish in under 2 minutes. My config is `--executor-memory 8g --executor-cores 4 --num-executors 15 --conf spark.sql.shuffle.partitions=200`. How do I fix this?

4. My job produces thousands of tiny output files (a few KB each) after a write to S3, and downstream jobs reading this data are slow. Current config uses 500 shuffle partitions on a 2GB dataset. How should I reduce the small-files problem?

5. My job has too few partitions — only 4 tasks are running even though I have `--num-executors 20 --executor-cores 4` (80 total cores available). What config change fixes the underutilization?

6. I'm seeing my job spill heavily to disk during shuffle, with `ShuffleWriteMetrics` showing large "bytes spilled." Current config: `--executor-memory 4g --executor-cores 8`. What should I adjust?

7. My broadcast join is timing out with `Could not execute broadcast in 300 secs`. The smaller table is around 900MB. Current config uses default `spark.sql.autoBroadcastJoinThreshold`. How do I fix this?

8. I have a small lookup table (under 50MB) being joined with a huge fact table via a regular sort-merge join, and it's slow. How do I force Spark to use a broadcast join instead?

9. My job with `spark.dynamicAllocation.enabled=true` keeps adding and removing executors rapidly, causing instability. What settings control this thrashing?

10. I'm seeing frequent full GC pauses (several seconds each) on my executors under heavy load. Current config: `--executor-memory 16g --executor-cores 8`. How do I reduce GC pressure?

11. My job caches a DataFrame with `.cache()` but I'm running low on executor memory and seeing evictions. How should I tune caching vs execution memory?

12. I'm re-computing the same DataFrame transformation multiple times across my pipeline without caching it, and it's slow. What's the right way to persist it?

13. My wide transformation (multiple joins + groupBy) is causing an explosion in shuffle data volume. How do I restructure the query or config to reduce shuffle size?

14. My job uses `.checkpoint()` on a DataFrame with a very long lineage, and checkpointing itself is taking a long time. What's the tradeoff here and how should I tune it?

15. I'm getting `FetchFailedException: Failed to connect` errors during a large shuffle fetch between executors. What network/timeout settings should I adjust?

16. My executors are being killed by YARN with `Container killed by YARN for exceeding memory limits`, even though `--executor-memory` looks sufficient. What's missing from my config?

17. My groupBy on a skewed key (one key has 80% of the rows) is creating one massive task. How do I handle this specific skew pattern?

18. My job's driver needs a lot of memory to plan a complex query with many joins, and I'm seeing driver OOM before any executor work starts. What driver-side settings help?

19. I have idle executors sitting around doing nothing most of the time, wasting cluster resources. Current config: `--num-executors 50 --executor-cores 2 --executor-memory 4g`. How do I right-size this?

20. My job is slow because parallelism is too low relative to available cores — only using `--executor-cores 1 --num-executors 5` on a large cluster. What should I change?

21. I haven't enabled off-heap memory and I'm seeing heap exhaustion during heavy shuffle. How do I configure off-heap memory correctly?

22. I forgot to enable Kryo serialization and I suspect serialization overhead is slowing down my job. How do I enable it and what else should I check?

23. My job doesn't have Adaptive Query Execution (AQE) enabled and I'm seeing poor join strategy choices and static partition counts. How do I turn AQE on and what does it help with?

---

## Data pipeline failures / performance

24. My daily batch Spark job normally runs in 20 minutes but is now taking 90 minutes without failing outright. Downstream reports are blocked. How do I diagnose this?

25. A Spark job that used to complete reliably now intermittently fails with executor loss, but there's no obvious error in the driver logs. Where do I start investigating?

26. My ETL pipeline's runtime has been growing steadily over the past few weeks even though data volume looks stable. What could cause this kind of gradual degradation?

27. I have a job that reads from a partitioned S3 dataset but seems to be scanning far more data than expected. How do I check if partition pruning is working?

---

## Streaming / Kafka

28. My Kafka consumer is falling behind and lag keeps growing even though I haven't changed anything. What should I check first?

29. My Spark Structured Streaming job is experiencing frequent micro-batch delays and increasing backpressure. How do I diagnose the bottleneck?

30. I'm seeing duplicate records downstream from my Kafka-based streaming pipeline. What are the common causes and how do I fix exactly-once semantics?

---

## Data quality

31. Our data quality checks started failing overnight — null rates on a key column spiked from under 1% to over 30%. How do I trace this back to the source?

32. I'm seeing row counts in a downstream table that don't match the expected counts from the source table after a join. What's the systematic way to debug this?

33. A schema validation step is rejecting records that used to pass. How do I figure out what changed upstream?

---

## Cloud cost / resources

34. Our monthly cloud compute costs jumped significantly after a recent pipeline change, even though data volume didn't grow much. How do I find what's driving the cost increase?

35. I have several long-running clusters that seem to be underutilized most of the time. What's a good strategy to reduce idle cluster cost?

---

## Orchestration / scheduling

36. My Airflow DAG has started missing its SLA and tasks are queuing up behind each other. How do I identify the bottleneck task?

37. A scheduled job occasionally fails to trigger at all, with no error logged. What are common causes of silently missed schedules?

---

## Schema evolution

38. A producer team changed a field type in our upstream data without notice, and it's now breaking our downstream schema. How should we have caught this earlier, and how do we handle it now?

39. We need to add a new nullable column to an existing Delta table without breaking existing readers. What's the safe way to do this?

---

## Backfills / reprocessing

40. We need to backfill three months of historical data through our pipeline, but running it all at once risks overwhelming downstream systems. What's a safe backfill strategy?

41. A bug in our transformation logic corrupted two weeks of data before we caught it. What's the process for identifying affected records and safely reprocessing them?

---

## Security / access

42. A teammate lost access to a cluster after an IAM policy update, but we're not sure exactly which permission is missing. How do we debug access issues systematically?

43. We need to rotate a secret used by multiple Spark jobs without causing downtime. What's the safe rotation process?

---

## Stakeholder / process pressure

44. Leadership wants an ETA on a data quality fix, but the root cause isn't fully understood yet. How do I communicate this without overpromising?

45. Two teams disagree on whose pipeline introduced a data inconsistency, and both point to the other. How do I approach resolving this productively?

---

## General / mixed (should retrieve across categories)

46. What's the difference between narrow and wide transformations in Spark, and why does it matter for performance?

47. How do I decide the right number of shuffle partitions for a given dataset size?

48. My Spark job works fine locally but fails only in the cluster environment. What's different that I should check?

49. What's a reasonable executor-to-core-to-memory ratio to start with for a general-purpose Spark job?

50. How do I tell whether a slow Spark job is CPU-bound, memory-bound, or I/O-bound?

---

## How to use this

- Paste each question into the Streamlit chat (`streamlit run app.py`) one at a time
- Check: did the retrieved documents actually match the topic? (expand "Retrieved documents")
- Check: did self-check flag anything, and if so, was the repair actually better?
- Use the sidebar component filter on a few questions to confirm filtering works
- Give thumbs up/down feedback on each — this populates the Monitoring dashboard with real data
- After running 15-20 of these, check the Monitoring page to see the dashboard populate
