

import unittest
import struct
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stdf_parser import STDFV4Parser


class TestSTDFParser(unittest.TestCase):

    def _create_minimal_stdf(self, filepath):
        """Create a minimal valid STDF file for testing"""
        with open(filepath, 'wb') as f:
            # FAR record: rec_len=2, rec_typ=0, rec_sub=10
            f.write(struct.pack('<H', 2))       # rec_len
            f.write(struct.pack('B', 0))        # rec_typ
            f.write(struct.pack('B', 10))       # rec_sub
            f.write(struct.pack('B', 2))        # cpu_type (Intel LE)
            f.write(struct.pack('B', 4))        # stdf_ver (V4)

            # MIR record (minimal)
            mir_data = bytearray()
            mir_data += struct.pack('<I', 0)    # setup_t
            mir_data += struct.pack('<I', 0)    # start_t
            mir_data += struct.pack('B', 1)     # stat_num
            mir_data += b'P'                    # mode_cod
            mir_data += b' '                    # rtst_cod
            mir_data += b' '                    # prot_cod
            mir_data += struct.pack('<H', 0)    # burn_tim
            mir_data += b' '                    # cmod_cod
            mir_data += struct.pack('B', 7) + b'TEST123'  # lot_id
            mir_data += struct.pack('B', 4) + b'CHIP'     # part_typ
            mir_data += struct.pack('B', 5) + b'NODE1'    # node_nam
            mir_data += struct.pack('B', 0)     # tstr_typ
            mir_data += struct.pack('B', 0)     # job_nam

            f.write(struct.pack('<H', len(mir_data)))
            f.write(struct.pack('B', 1))
            f.write(struct.pack('B', 10))
            f.write(mir_data)

            # PTR record
            ptr_data = bytearray()
            ptr_data += struct.pack('<I', 100)          # test_num
            ptr_data += struct.pack('B', 0)             # head_num
            ptr_data += struct.pack('B', 0)             # site_num
            ptr_data += struct.pack('B', 0)             # test_flg (pass)
            ptr_data += struct.pack('B', 0)             # parm_flg
            ptr_data += struct.pack('<f', 3.3)          # result
            ptr_data += struct.pack('B', 8) + b'VDD_TEST'  # test_txt
            ptr_data += struct.pack('B', 0)             # alarm_id
            ptr_data += struct.pack('B', 0)             # opt_flag
            ptr_data += struct.pack('b', 0)             # res_scal
            ptr_data += struct.pack('b', 0)             # llm_scal
            ptr_data += struct.pack('b', 0)             # hlm_scal
            ptr_data += struct.pack('<f', 3.0)          # lo_limit
            ptr_data += struct.pack('<f', 3.6)          # hi_limit
            ptr_data += struct.pack('B', 1) + b'V'     # units

            f.write(struct.pack('<H', len(ptr_data)))
            f.write(struct.pack('B', 15))
            f.write(struct.pack('B', 10))
            f.write(ptr_data)

    def test_parse_minimal_stdf(self):
        with tempfile.NamedTemporaryFile(suffix='.stdf', delete=False) as tmp:
            self._create_minimal_stdf(tmp.name)

            parser = STDFV4Parser(tmp.name)
            parser.parse()

            self.assertEqual(len(parser.ptr_records), 1)
            self.assertEqual(parser.ptr_records[0].test_num, 100)
            self.assertAlmostEqual(parser.ptr_records[0].result, 3.3, places=1)
            self.assertEqual(parser.ptr_records[0].test_txt, 'VDD_TEST')

            self.assertIsNotNone(parser.mir)
            self.assertEqual(parser.mir.lot_id, 'TEST123')

            os.unlink(tmp.name)

    def test_dataframe_export(self):
        with tempfile.NamedTemporaryFile(suffix='.stdf', delete=False) as tmp:
            self._create_minimal_stdf(tmp.name)

            parser = STDFV4Parser(tmp.name)
            parser.parse()

            df = parser.get_ptr_dataframe(site=0)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['test_name'], 'VDD_TEST')
            self.assertEqual(df.iloc[0]['pass_fail'], 'PASS')

            os.unlink(tmp.name)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            STDFV4Parser("nonexistent_file.stdf")


if __name__ == '__main__':
    unittest.main()
