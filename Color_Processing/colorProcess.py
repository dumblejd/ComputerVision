def BGR2LUV(b, g, r):
    # no linear
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    # linear rgb
    b = invgamma(b)
    g = invgamma(g)
    r = invgamma(r)
    # XYZ linear
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    # LUV
    L, u, v = XYZ2LUV(X, Y, Z)
    return [L, u, v]


def BGR2xyY(b, g, r):
    # no linear
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    # linear rgb
    b = invgamma(b)
    g = invgamma(g)
    r = invgamma(r)
    # XYZ linear
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    return x, y, Y


def LUV2BGR(L, u, v):
    # XYZ
    X, Y, Z = LUV2XYZ(L, u, v)
    # sRGB
    Rsrgb = 3.240479 * X - 1.53715 * Y - 0.498535 * Z
    Gsrgb = -0.969256 * X + 1.875991 * Y + 0.041556 * Z
    Bsrgb = 0.055648 * X - 0.204043 * Y + 1.057311 * Z

    r = gamma(Rsrgb)
    g = gamma(Gsrgb)
    b = gamma(Bsrgb)

    if r > 1:
        r = 1.0
    elif r < 0:
        r = 0.0
    if g > 1:
        g = 1.0
    elif g < 0:
        g = 0.0
    if b > 1:
        b = 1.0
    elif b < 0:
        b = 0.0
    r = round(r * 255.0)
    g = round(g * 255.0)
    b = round(b * 255.0)

    return [b, g, r]


def linearScaling(x, min, max, toMin, toMax):
    x = (x - min) * (toMax - toMin) * 1.0 / (max - min) + toMin
    return x


def gamma(x):
    if x < 0.00304:
        x = 12.92 * x
    else:
        x = (1.055 * (x ** (1.0 / 2.4)) - 0.055)
    return x


def invgamma(x):
    if x < 0.03928:
        x = x / 12.92
    else:
        x = ((x + 0.055) / 1.055) ** 2.4
    return x


def LUV2XYZ(L, u, v):
    xw = 0.95
    yw = 1.0
    zw = 1.09
    uw = 4 * xw / (xw + 15 * yw + 3 * zw)
    vw = 9 * yw / (xw + 15 * yw + 3 * zw)
    if L == 0:
        return [0, 0, 0]
    uprime = (u + 13 * uw * L) / (13 * L)
    vprime = (v + 13 * vw * L) / (13 * L)

    if L > 7.9996:
        Y = (((L + 16) / 116.0) ** 3) * yw
    else:
        Y = (L * yw / 903.3)
    if vprime == 0:
        return [0, Y, 0]
    else:
        X = (2.25 * Y * uprime) / vprime
        Z = Y * (3.0 - 0.75 * uprime - 5 * vprime) / vprime
        return [X, Y, Z]


def XYZ2LUV(x, y, z):
    xw = 0.95
    yw = 1.0
    zw = 1.09
    uw = 4 * xw / (xw + 15 * yw + 3 * zw)
    vw = 9 * yw / (xw + 15 * yw + 3 * zw)
    u = 0.0
    v = 0.0
    t = y / yw
    if t > 0.008856:
        L = 116 * (t ** (1.0 / 3.0)) - 16
    else:
        L = 903.3 * t

    d = x + 15 * y + 3 * z
    if (d != 0):
        uprime = 4 * x / d
        vprime = 9 * y / d
        u = 13 * L * (uprime - uw)
        v = 13 * L * (vprime - vw)
    return [L, u, v]


def xyY2LUV(x, y, Y):
    X = x * 1.0 / y * Y
    Z = (1 - x - y) / y * Y
    L, u, v = XYZ2LUV(X, Y, Z)
    return L, u, v
