import argparse,logging,time
import mxnet as mx
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)

def separable_conv(data, n_in_ch, n_out_ch, kernel, pad, stride, name, depth_mult=1):
	#  depthwise convolution
	#channels = mx.sym.SliceChannel(data=data, axis=1, num_outputs=n_in_ch)
	#dw_outs = [mx.sym.Convolution(data=channels[i], num_filter=depth_mult, pad=pad, kernel=kernel, stride=stride, no_bias=True, name=name+'_depthwise_kernel'+str(i)) for i in range(n_in_ch)]
	#dw_out = mx.sym.Concat(*dw_outs)
	dw_out = mx.sym.Convolution(data=data, num_filter=n_in_ch, num_group=n_in_ch, kernel=kernel, stride=stride, pad=pad, no_bias=True, name=name + '_depthwise_kernel')
	dw_out = mx.sym.BatchNorm(data=dw_out, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'/3x3/bn')
	dw_out = mx.sym.Activation(data=dw_out, act_type='relu', name=name+'/3x3/relu')
	#  pointwise convolution
	pw_out = mx.sym.Convolution(data=dw_out, num_filter=n_out_ch, kernel=(1, 1), no_bias=True, name=name+'_pointwise_kernel')
	pw_out = mx.sym.BatchNorm(data=pw_out, fix_gamma=False, eps=2e-5, momentum=0.9, name=name+'/1x1/bn')
	pw_out = mx.sym.Activation(data=pw_out, act_type='relu', name=name+'/1x1/relu')
	return pw_out

def MobileNet_feature(alpha):
	data = mx.sym.Variable('data')
	data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
	b1 = mx.sym.Convolution(data=data, num_filter=int(32*alpha), kernel=(3,3), stride=(2,2), pad=(1,1), no_bias=True, name='conv1')
	b1 = mx.sym.BatchNorm(data=b1, fix_gamma=False, eps=2e-5, momentum=0.9, name='conv1/bn')
	b1 = mx.sym.Activation(data=b1, act_type='relu', name='conv1/relu')
	b2 = separable_conv(b1, int(32*alpha), int(64*alpha), (3,3), (1,1), (1,1), 'block2_sepconv1')
	b3 = separable_conv(b2, int(64*alpha), int(128*alpha), (3,3), (1,1), (2,2), 'block3_sepconv1')
	b4 = separable_conv(b3, int(128*alpha), int(128*alpha), (3,3), (1,1), (1,1), 'block4_sepconv1')
	b5 = separable_conv(b4, int(128*alpha), int(256*alpha), (3,3), (1,1), (2,2), 'block5_sepconv1')
	b6 = separable_conv(b5, int(256*alpha), int(256*alpha), (3,3), (1,1), (1,1), 'block6_sepconv1')
	b7 = separable_conv(b6, int(256*alpha), int(512*alpha), (3,3), (1,1), (2,2), 'block7_sepconv1')
	b = b7
	for i in range(5):
		prefix = 'block' + str(i + 8)
		b = separable_conv(b, int(512*alpha), int(512*alpha), (3,3), (1,1),(1,1), prefix+'_sepconv1')
	b13 = separable_conv(b, int(512*alpha), int(1024*alpha), (3,3), (1,1), (2,2), 'block13_sepconv1')
	b14 = separable_conv(b13, int(1024*alpha), int(1024*alpha), (3,3), (1,1), (2,2), 'block14_sepconv1')
	pool = mx.sym.Pooling(b14, kernel=(4,4), global_pool=True, pool_type='avg', name='global_pool')
	#drop = mx.symbol.Dropout(data=pool, p=0.5, name="drop1")
	return pool

def MobileNet(alpha):
	fea = MobileNet_feature(alpha)
	flat = mx.symbol.Flatten(data=fea)
	fc = mx.symbol.FullyConnected(data=flat, num_hidden=1000, name='predictions')
	softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
	return softmax

def main():
	mx.random.seed(0)
	if args.log_file:
		fh = logging.FileHandler(args.log_file)
		logger.addHandler(fh)
	MobileNetSymbol = MobileNet(alpha=0.5)
	devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
	epoch_size = args.num_examples / args.batch_size
	print(epoch_size)
	checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
	kv = mx.kvstore.create(args.kv_store)
	arg_params = None
	aux_params = None
	if args.retrain:
		_, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)

	train = mx.io.ImageRecordIter(
		path_imgrec = "/path/to/rec",
		data_shape  = (1, 128, 128),
		scale       = 1./255,
		batch_size  = args.batch_size,
		rand_crop   = True,
		rand_mirror = True,
		num_parts   = kv.num_workers,
		part_index  = kv.rank)
	val = None
	model = mx.model.FeedForward(
		ctx                = devs,
		symbol             = MobileNetSymbol,
		arg_params         = arg_params,
		aux_params         = aux_params,
		#optimizer          = 'rmsprop',
		#clip_gradient      = 1.0,
		num_epoch          = 200,
		learning_rate      = args.lr,
		momentum           = 0.9,
		#wd                 = 0.0000005,
		wd                 = 0.0005,
		#lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=5*max(int(epoch_size * 1), 1), factor=0.8, stop_factor_lr=5e-6),
		#lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=5*max(int(epoch_size * 1), 1), factor=0.1, stop_factor_lr=5e-5),
		lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=50*max(int(epoch_size * 1), 1), factor=0.5, stop_factor_lr=5e-6),
		initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))
	eval_metrics = mx.metric.CompositeEvalMetric()
	eval_metrics.add(mx.metric.SoftMax_Loss())
	eval_metrics.add(mx.metric.create('acc'))
	model.fit(
		X                  = train,
		eval_data          = val,
		kvstore            = kv,
		eval_metric       = eval_metrics,
		batch_end_callback = mx.callback.Speedometer(args.batch_size, 100),
		epoch_end_callback = checkpoint)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="command for training imagenet")
	parser.add_argument('--gpus', type=str, default='1', help='the gpus will be used, e.g "0,1,2,3"')
	parser.add_argument('--model_save_prefix', type=str, default='./models/MobileNet1', help='the prefix of the model to save')
	#0.05
	parser.add_argument('--lr', type=float, default=0.05, help='initialization learning reate')
	#2048
	parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
	parser.add_argument('--num_examples', type=int, default=370783, help='the number of training examples')
	parser.add_argument('--kv_store', type=str, default='local', help='the kvstore type')
	parser.add_argument('--model_load_prefix', type=str, default='./models/MobileNet1', help='the prefix of the model to load')
	parser.add_argument('--model_load_epoch', type=int, default=1, help='load the model on an epoch using the model_load_prefix')
	parser.add_argument('--log_file', type=str, default="train_MobileNet1.log", help='save training log to file')
	parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
	args = parser.parse_args()
	logging.info(args)
	main()
