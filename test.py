import ultralytics.nn.modules.head
from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, ROOT, yaml_load
from torchsummary import summary

FCN = ultralytics.nn.modules.head.DetectFCN


