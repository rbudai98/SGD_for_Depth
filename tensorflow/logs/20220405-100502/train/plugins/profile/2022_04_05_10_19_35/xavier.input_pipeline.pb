$	Y???l?"@?dz\R??CSv?AM!@!ɒ9?w5$@$	1>Hn?e @  v?? @`? g?@!?2m??'@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????#@R?o&??@A"??p?@Y?!S>???r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1tE)!Xm#@?0?????A?:?? ?@Y??_ ??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Yf?="@?y??Q? @A ?_>Y?@Y]?mO????r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1a5??V"@??#??@AHlw?m@Y?W?\??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1ɒ9?w5$@d??A%@A*??s?@Ykf-????r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?yȔ?#@,??26@A???KUZ@Y?J %v???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?????"@??O8????A?^???&@Y????O???r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?/?'$@ }??A?@AF#?W<@Y?k$	???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??E;?!@?L?D????AL???<?@YB?"LQ.??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1r?@H?!@??	???AU2 Tq?@Y????C??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??/???!@Z???-??A?;?*@Y???c???r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???VC?"@??-@{@AƿϸpP@Yqt???r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1LP÷?6"@ۤ??????A~?$A??@YO?\???r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1$*T7?!@? ??E??A9?j?3?@Y?o?????r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1? [???!@?(?QGG@A???o^?@Y?j??????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1CSv?AM!@??"ڎ???A??v??:@YKi?,??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1? ?b?!@}>ʈ???A?g?u??@Y%!?????r	train 517*	?rh?-??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatM.??:?@!??U*??A@)'?O:?`@19[Pw?@@:Preprocessing2T
Iterator::Root::ParallelMapV2!?J???!?-???2@)!?J???1?-???2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice 	?v???!E??H?.+@) 	?v???1E??H?.+@:Preprocessing2E
Iterator::Root????_ @!?l?k)?@)>&R?????1?~꛼?(@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W?????!Cx?/??5@)???h??1A@??? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?t?@!?d%?5Q@)f??CC??1??,?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??9????!?F?7??9@)??9????1?t???i@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*erjg????!?V0??@)erjg????1?V0??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t22.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9+?s? L @I??Q??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	.----m @?츒?w??}>ʈ???!??#??@	!       "	!       *	!       2$	k????@??ը?Q??Hlw?m@!*??s?@:	!       B	!       J$	??G%:??l?]/Wz??kf-????!?o?????R	!       Z$	??G%:??l?]/Wz??kf-????!?o?????b	!       JCPU_ONLYY+?s? L @b q??Q??V@