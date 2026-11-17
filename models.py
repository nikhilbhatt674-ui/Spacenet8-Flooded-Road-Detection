"""ResNet-34 U-Net variants and losses. Only torch + torchvision are required."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResNet34Encoder(nn.Module):
    """Returns 5 feature maps at strides 2,4,8,16,32 with channels 64,64,128,256,512."""
    channels = [64, 64, 128, 256, 512]

    def __init__(self, pretrained=True, in_ch=3):
        super().__init__()
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet34(weights=w)
        if in_ch != 3:  # early fusion: tile the pretrained RGB filters over both dates, rescale
            conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            conv.weight.data = m.conv1.weight.data.repeat(1, in_ch // 3, 1, 1) / (in_ch // 3)
            m.conv1 = conv
        self.conv1, self.bn1, self.pool = m.conv1, m.bn1, m.maxpool
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        f0 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        f1 = self.l1(self.pool(f0)); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        return [f0, f1, f2, f3, f4]


class Decoder(nn.Module):
    def __init__(self, skip_ch, dec_ch=(256, 128, 64, 32), out_ch=16):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = skip_ch[4]
        for lvl, oc in zip((3, 2, 1, 0), dec_ch):
            self.blocks.append(conv_block(prev + skip_ch[lvl], oc)); prev = oc
        self.final = conv_block(prev, out_ch); self.out_ch = out_ch

    def forward(self, feats, size):
        x = feats[4]
        for blk, lvl in zip(self.blocks, (3, 2, 1, 0)):
            x = F.interpolate(x, size=feats[lvl].shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, feats[lvl]], 1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.final(x)


class FloodNet(nn.Module):
    """mode: 'post' | 'early' | 'siam_diff' | 'siam_nodiff'.  road_head: auxiliary road-prior branch."""

    def __init__(self, mode="siam_diff", road_head=True, pretrained=True):
        super().__init__()
        self.mode, self.has_road = mode, road_head
        self.enc = ResNet34Encoder(pretrained, in_ch=6 if mode == "early" else 3)
        mult = {"post": 1, "early": 1, "siam_nodiff": 2, "siam_diff": 3}[mode]
        self.dec = Decoder([c * mult for c in self.enc.channels])
        self.flood_head = nn.Conv2d(self.dec.out_ch, 3, 1)
        self.road_head = nn.Conv2d(self.dec.out_ch, 1, 1) if road_head else None

    def forward(self, pre, post):
        size = post.shape[-2:]
        if self.mode == "post":
            feats = self.enc(post)
        elif self.mode == "early":
            feats = self.enc(torch.cat([pre, post], 1))
        else:  # shared weights: same encoder applied to both dates (separate forward passes)
            fa, fb = self.enc(pre), self.enc(post)
            if self.mode == "siam_diff":
                feats = [torch.cat([a, b, (a - b).abs()], 1) for a, b in zip(fa, fb)]
            else:
                feats = [torch.cat([a, b], 1) for a, b in zip(fa, fb)]
        h = self.dec(feats, size)
        return dict(flood=self.flood_head(h), road=self.road_head(h) if self.has_road else None)


# ---------------------------------------------------------------- losses
def focal_loss(logits, y, alpha, gamma=2.0):
    logp = F.log_softmax(logits.float(), 1)
    logpt = logp.gather(1, y[:, None])[:, 0]
    return -(alpha[y] * (1 - logpt.exp()).pow(gamma) * logpt).mean()


def dice_loss(logits, y, cls=2, eps=1.0):
    p = F.softmax(logits.float(), 1)[:, cls]; g = (y == cls).float()
    return 1 - (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def total_loss(out, y, road_y, kind, alpha, gamma=2.0, lam=1.0, mu=0.5):
    if kind == "ce":
        l = F.cross_entropy(out["flood"].float(), y, weight=alpha)
    else:
        l = focal_loss(out["flood"], y, alpha, gamma)
        if kind == "focal_dice":
            l = l + lam * dice_loss(out["flood"], y)
    if out["road"] is not None:
        l = l + mu * F.binary_cross_entropy_with_logits(out["road"][:, 0].float(), road_y.float())
    return l
