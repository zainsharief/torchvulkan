implementing dtype;

// bool runs natively as uint8 (kernels normalise it where they read values). Without 8-bit storage it is
// emulated: any nonzero byte reads as 1 and every store writes 0 or 1
public struct BoolCodec : IIntCodec
{
    public static uint bytes() { return 1; }
    public static int decode(uint bits) { return bits != 0 ? 1 : 0; }
    public static uint encode(int value) { return value != 0 ? 1 : 0; }
    public static int lowest() { return 0; }
    public static int highest() { return 1; }
}

public typealias BoolEmulated = EmulatedInt<BoolCodec>;
