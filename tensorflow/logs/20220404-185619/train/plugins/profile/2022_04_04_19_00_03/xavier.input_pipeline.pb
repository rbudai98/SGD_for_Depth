$	?? $S+@3??CO@X??%@!֍wG??2@$	?v??Yd'@?	?8? @??i?j@!ڋ?fC@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1X??%@z??C5e@A^I?\?@YjK?????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??[?T&@?GĔ?@A?N^d @Y\ A?c???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??,AF?%@_?D??@A??Ϝ??@Y?^???r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1t????'@??w}??@A?X???!@Yx??Dg???r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?聏?b-@?I?%r?@A?\??X$@YX˝?`???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1S?r/0C0@??8???@A?c?i)@Y?jI??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	??Q??+@F???j?@A???1vB!@Y?M)??P??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
}?ݮ??*@,?,?@AvT5A??"@Y???\Q???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1֍wG??2@KY?8օ@AA)Z?p-@Y???;?_??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Va3?1.@??8?Վ??Ayv??_$@Y?BW"P?@r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1<?l?V+@???l??@A¿3i@Y??mē?@r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?g??/@??ᔹ???A?6?Ӂ?'@Yk???#?@r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1 y?P??/@?شRd??Aa?4??+@Y??Gq??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????c?'@?o&?1??Aę_?"@Y'?;z??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?k???'@vOjM@A?K???@Y?+?F<???r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1k*??..@5?b??^@A?cϞK$@Yo?;2V? @r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???8+?'@8?T??@A?|?5^?@Y?{?ڥ??r	train 517*	?S?????@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??w??D@!&_???@@)??ky?@1??һ=@:Preprocessing2T
Iterator::Root::ParallelMapV2??>9
P??!??TKB?5@)??>9
P??1??TKB?5@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?yY???!?????-@)?yY???1?????-@:Preprocessing2E
Iterator::Root:??i@!H?u}z?A@)??Ơ???1??-_e?*@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???M?\@!?!E?B,P@)?????j??1Td?0F6 @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatenj???;??!2?E^|?5@)8??P??1??L?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*M?^?iN??!?;?q`	@)M?^?iN??1?;?q`	@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???? @!?? `8@)?k%t????19ޠj@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Ir???k'@I?Q?͆V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??+|x@<???PO???o&?1??!F???j?@	!       "	!       *	!       2$	IV ?#@?>?Z?+@¿3i@!A)Z?p-@:	!       B	!       J$	??W??????!???\ A?c???!??mē?@R	!       Z$	??W??????!???\ A?c???!??mē?@b	!       JCPU_ONLYYIr???k'@b q?Q?͆V@