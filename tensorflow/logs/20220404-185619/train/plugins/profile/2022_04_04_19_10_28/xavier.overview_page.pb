?&$	C??p?!@?H~1??~?p?? @!?SHޙ#@$	1??֦?@@ ?C1?@@???????!Cj?ˆ	)@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?GS=?"@?"j??G @A?V'g(?@Y?v?$$???r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????{j#@?}?<?@A?P???@Y!??nJ??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??QI??"@5?Ry;???A?^~???@Yڐf??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?/?ۆ!@?4?B???Au???@Y???????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1W?SbZ"@G?&ji? @AR?????@Yy???e???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1[??Y!@?Op??F??A??ިv@Y]?wb???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	˟o?!@㊋????Ag?v??@Y??eN????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
~?p?? @uʣaQ??A?St$?_@Y噗??;??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Hڍ>? @g???p???A?j?0?@YQL? 3??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?3??!@?q?_a??A/?
Ҍ?@Y?"j??G??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?ْU?"@2???M??A?3???@Y[? m?Y??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????"@?$$?6~??AW\?{@Y[(?????r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?
Y?? @???D????Aմ?i?+@Y??vö??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?SHޙ#@??? e??A-&?(?@Yn?r???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?3?ތ? @???????A???? @Y? Q0c
??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?fd??x!@?!6X8???AQ??le@Y?????M??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?B??K"@Y?E?????A?*???@Y<??ؖ??r	train 517*	?G?Zϻ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?[?J?@!H3%<??@@)I??_?? @1?!?xk?<@:Preprocessing2E
Iterator::Root)????H@!?`??j?B@)\?=????1??z?834@:Preprocessing2T
Iterator::Root::ParallelMapV2?$?9???!T0Ҝ+1@)?$?9???1T0Ҝ+1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceZK ????!뛎g?))@)ZK ????1뛎g?))@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip? ????@!?<?PO@)B#ظ?]??1?'??? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR&5???!?)ɔ??2@)??=?
??1o???@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?2SZK??!??S?@)?2SZK??1??S?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap:??KT??!?C3Ͽ[5@)???O???1??P?)?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9]DBߚ@I??)X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?C?sA??#??%A??????????!?}?<?@	!       "	!       *	!       2$	?2????@?(????????ިv@!W\?{@:	!       B	!       J$	}̵???tU	???[(?????!]?wb???R	!       Z$	}̵???tU	???[(?????!]?wb???b	!       JCPU_ONLYY]DBߚ@b q??)X@Y      Y@qփ?(??@"?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 