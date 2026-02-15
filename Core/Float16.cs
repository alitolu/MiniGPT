using System;

namespace MiniGPT.Core
{
    public struct Float16
    {
        public ushort Bits;

        public Float16(float f)
        {
            Bits = FloatToHalf(f);
        }

        public float ToFloat()
        {
            return HalfToFloat(Bits);
        }

        static ushort FloatToHalf(float f)
        {
            uint x = BitConverter.ToUInt32(
                BitConverter.GetBytes(f),0);

            uint sign = (x >> 16) & 0x8000;
            uint mant = x & 0x007fffff;
            int exp = (int)((x >> 23) & 0xff) - 127 + 15;

            if (exp <= 0) return (ushort)sign;
            if (exp >= 31) return (ushort)(sign | 0x7c00);

            return (ushort)(sign | ((uint)exp << 10) | (mant >> 13));
        }

        static float HalfToFloat(ushort h)
        {
            uint sign = (uint)(h & 0x8000) << 16;
            uint exp = (uint)(h & 0x7C00) >> 10;
            uint mant = (uint)(h & 0x03FF);

            if (exp == 0)
                return BitConverter.ToSingle(
                    BitConverter.GetBytes(sign),0);

            exp = exp + (127 - 15);

            uint result =
                sign |
                (exp << 23) |
                (mant << 13);

            return BitConverter.ToSingle(
                BitConverter.GetBytes(result),0);
        }
    }
}
