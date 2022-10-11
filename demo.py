
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class RotConv_circle(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        assert self.kernel_size == (3,3)

        # '''CIRCLE'''
        # _weight = torch.ones_like(self.weight)
        # _a = self.weight[:,:,0,0]
        # _b = self.weight[:,:,0,1]
        # _sq_2 = 1.414213562

        # _weight[:,:,0,0] = 0.5*_b
        # _weight[:,:,2,0] = 0.5*_b
        # _weight[:,:,0,2] = 0.5*_b
        # _weight[:,:,2,2] = 0.5*_b
        # _weight[:,:,0,1] = _sq_2*_b
        # _weight[:,:,1,0] = _sq_2*_b
        # _weight[:,:,1,2] = _sq_2*_b
        # _weight[:,:,2,1] = _sq_2*_b
        # _weight[:,:,1,1] = _a + (6 - 4*_sq_2)*_b


#         '''CIRCLE Integral'''
#         _weight = torch.ones_like(self.weight)
#         _a = self.weight[:,:,0,0]
#         _b = self.weight[:,:,0,1]
#         _c = self.weight[:,:,0,2]
#         _d = self.weight[:,:,1,0]
#         _e = self.weight[:,:,1,1]
#         _f = self.weight[:,:,1,2]
#         _g = self.weight[:,:,2,0]
#         _h = self.weight[:,:,2,1]
#         _i = self.weight[:,:,2,2]

#         rb = 1/8
#         rc = 2/8
#         rd = 3/8
#         re = 4/8
#         rf = 5/8
#         rg = 6/8
#         rh = 7/8
#         ri = 8/8
    
#         _pi = 3.14159265357

#         _t = 0.5*(rb*rb*_b + rc*rc*_c + rd*rd*_d + re*re*_e + rf*rf*_f + rg*rg*_g + rh*rh*_h + ri*ri*_i)

#         _m = (2*rb-rb*rb)*_b + (2*rc-rc*rc)*_c + (2*rd-rd*rd)*_d + (2*re-re*re)*_e + \
#         (2*rf-rf*rf)*_f + (2*rg-rg*rg)*_g + (2*rh-rh*rh)*_h + (2*ri-ri*ri)*_i

#         _n = _a + (-8*rb+2*rb*rb)*_b + (-8*rc+2*rc*rc)*_c + (-8*rd+2*rd*rd)*_d + (-8*re+2*re*re)*_e + \
#         (-8*rf+2*rf*rf)*_f + (-8*rg+2*rg*rg)*_g + (-8*rh+2*rh*rh)*_h + (-8*ri+2*ri*ri)*_i + \
#         (_b + _c + _d + _e + _f + _g + _h + _i)*2*_pi
        
#         _weight[:,:,0,0] = _t
#         _weight[:,:,2,0] = _t
#         _weight[:,:,0,2] = _t
#         _weight[:,:,2,2] = _t
#         _weight[:,:,0,1] = _m
#         _weight[:,:,1,0] = _m
#         _weight[:,:,1,2] = _m
#         _weight[:,:,2,1] = _m
#         _weight[:,:,1,1] = _n

       
        #单圆弧
        _weight = torch.ones_like(self.weight)
        _a = self.weight[:,:,0,0]
        _b = self.weight[:,:,0,1]
    
        _pi = 3.14159265357

        _t = 0.5*_b

        _m = _b

        _n = _a + (2*_pi-6)*_b 
        
        _weight[:,:,0,0] = _t
        _weight[:,:,2,0] = _t
        _weight[:,:,0,2] = _t
        _weight[:,:,2,2] = _t
        _weight[:,:,0,1] = _m
        _weight[:,:,1,0] = _m
        _weight[:,:,1,2] = _m
        _weight[:,:,2,1] = _m
        _weight[:,:,1,1] = _n

        x_features = self._conv_forward(input, _weight, self.bias)


        return x_features
