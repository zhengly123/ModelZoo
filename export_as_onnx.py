import torch
import torchvision.models as models
from shape_infer import infer_onnx
from src.csrnet import csrnet
from src.dcgan import _netG as dcgan_netG
from src.cgan import GeneratorCGAN as cgan_netG
from src.GhostNet import ghost_net
from src.scnet import scnet101
from src.hetconv import vgg16bn as hetconv_vgg16bn 
from src.pyconv.build_model import build_model as pyconv_model
from src.pyconv.args_file import parser as pyconv_args
from src.DC_CDN_IJCAI21 import build_CDCN
from src.net48 import Net48
from src.testnet import TestNet
from src.gcn_1703_02719 import GCN
from src.unet import UNet
# from src.wdsr_a import MODEL as WDSR
from src.super_resolution.main import arch as WDSR # git@github.com:achie27/super-resolution.git


batch_size = 1


def export_inferred_onnx(name, input_shape, precision=32):
    fn = f"onnx/{name}.bs{batch_size}.onnx"
    dummy_input = torch.randn(*input_shape)
    print(name, '='*20)
    if name == 'csrnet':
        model = csrnet()
    elif name == 'dcgan':
        model = dcgan_netG(1)
    elif name == 'cgan':
        z_dim, c_dim = 100, 10
        model = cgan_netG(z_dim=z_dim, c_dim=c_dim)
    elif name == 'ghost':
        model = ghost_net()
    elif name == 'scnet':
        model = scnet101()
    elif name == 'hetconv':
        n_parts=1
        model = hetconv_vgg16bn(64, 64, 128, 128, 256, 256, 256,
                            512, 512, 512, 512, 512, 512, n_parts)
    elif name == 'pyconv': # Pyramid conv
        model = pyconv_model(pyconv_args.parse_args(args=['-a','pyconvresnet']))
    elif name =='dc-cdn':
        model = build_CDCN()
    elif name =='Net48':
        model = Net48()
    elif name =='TestNet':
        model = TestNet()
    elif name =='GCN':
        model = GCN(21)
    elif name =='unet':
        model = UNet(3, 10, [32, 64, 32], 0, 3, group_norm=0)
    elif name in ['srcnn', 'fsrcnn', 'espcn', 'edsr', 'srgan', 'esrgan', 'prosr']:
        model = WDSR(name, 4, True).getModel()
    else:
        model = getattr(models, name)(pretrained=True)

    if precision == 16:
        model = model.eval().cuda().half()
        dummy_input = dummy_input.cuda().half()
        fn = f"onnx_fp16/{name}.onnx"
    else:
        model = model.eval().cuda()
        dummy_input = dummy_input.cuda()
    # # Without parameters
    # torch.onnx.export(model, dummy_input, fn,
    #   export_params=False, verbose=False)
    torch.onnx.export(model, dummy_input, fn,
                      export_params=True, verbose=False, opset_version=13)
                    #   export_params=True, verbose=False, opset_version=12)
    infer_onnx(fn)

    # Print model info
    with open(f'info/{name}.txt', 'w') as f:
        print(model, file=f)


# input_shape = (batch_size, 100, 1, 1)
# export_inferred_onnx('dcgan', input_shape)

# input_shape = (1, 100+10, 1, 1)
# export_inferred_onnx('cgan', input_shape, 'cgan.onnx')

# input_shape = (batch_size, 3, 224, 224)
# for name in ('resnet18', 'alexnet', 'squeezenet1_0', 'vgg16', 'densenet161', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0'):
#     export_inferred_onnx(name, input_shape)

# input_shape = (batch_size, 3, 299, 299)
# for name in ('inception_v3',):
#     export_inferred_onnx(name, input_shape)

# input_shape = (batch_size, 512, 14, 14)
# export_inferred_onnx('csrnet', input_shape)

# input_shape = (32,3,32,32)
# export_inferred_onnx('ghost', input_shape)

# input_shape = (batch_size, 3, 224, 224)
# export_inferred_onnx('scnet', input_shape)

# input_shape = (batch_size, 3, 14, 14)
# export_inferred_onnx('hetconv', input_shape)

# input_shape = (batch_size, 3, 224, 224)
# export_inferred_onnx('pyconv', input_shape)

# input_shape = (batch_size, 3, 224, 224)
# export_inferred_onnx('dc-cdn', input_shape)

# input_shape = (batch_size, 3, 48, 48)
# export_inferred_onnx('Net48', input_shape)

# input_shape = (batch_size, 256, 24, 24)
# export_inferred_onnx('TestNet', input_shape)

# input_shape = (1, 3, 224, 224)
# export_inferred_onnx('GCN', input_shape)

# input_shape = (batch_size, 3, 224, 224)
# export_inferred_onnx('regnet', input_shape)

# input_shape = (batch_size, 3, 224, 224)
# export_inferred_onnx('unet', input_shape)

# for name in ['srcnn', 'fsrcnn', 'espcn', 'edsr', 'srgan', 'esrgan', 'prosr']:
#     print(f'Generate {name}')
#     if name in ['srcnn', 'fsrcnn', 'espcn', 'srgan']:
#         input_shape = (batch_size, 1, 32, 32)
#     else:
#         input_shape = (batch_size, 3, 32, 32)
#     export_inferred_onnx(name, input_shape)
