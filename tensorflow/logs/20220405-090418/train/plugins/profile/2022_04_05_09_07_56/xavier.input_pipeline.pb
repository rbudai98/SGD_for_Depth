$	?????2(@C???r@	7U?#@!S>U?#3@$	?me%O?#@????O?@X(H??@!?rd???<@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1>???4?%@?????1@A???{!@Y?(z?c0??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails18-x?W?0@EF$a?@A?<i??)@Y)??????r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?*??<?&@ Ǟ=??@A??^???@Y;q9^?(??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1VW@?*@E,b?a,@A?	??.?#@YKw?ِ???r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1(I?L?Y$@?-</@A??;?_>@Y??>?<??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	7U?#@??1?M?@Apz??@Y??$???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	y?Z_&@?7?a??@A?e??@Y?ۄ{e???r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?U?p?%@??=???@A??V*@Y?@?9w???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1S>U?#3@??^
Z@AW??U-.@YRE?*k??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1i?(*@??6S!?@A?e?I)?!@Y??u????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1R<???(@x??qO@A???b? @Yi????^??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1^????(@?-t%?@A??3?I? @Y?aۢ???r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??:r?3(@uF@A,??!@YYR?>????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???[?%@-??x>?@A???=+@Y?M*k??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??q???$@???а?@A/R(_o@Y??]P_??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?6???$@1	?@AM??.?@Y?:M??@r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?"j???&@??g?@Arݔ?Z?@Yu;?ʃt??r	train 517*	????딿@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??\???@!V??@@)??X?@1*?TX?%=@:Preprocessing2T
Iterator::Root::ParallelMapV2?b? ̭??!???(?e5@)?b? ̭??1???(?e5@:Preprocessing2E
Iterator::Root??e?O?@!????JC@)?L??????1??????0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{C???!e???.@){C???1e???.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???M?@!	BKx??N@)-%?I(???1ɯ	?[@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateϾ? =??!B???W3@)??5?K??1?%?zTa@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???4)??!???@)???4)??1???@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapE???J???!t/:(M6@)????n???1??a???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t22.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9T#?6#@I~?;i<?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	x?j?m?@?-ę9????1?M?@!-??x>?@	!       "	!       *	!       2$	-4v(?j @?rN%?@M??.?@!W??U-.@:	!       B	!       J$	??Dxq???Ps?-`???(z?c0??!?:M??@R	!       Z$	??Dxq???Ps?-`???(z?c0??!?:M??@b	!       JCPU_ONLYYT#?6#@b q~?;i<?V@