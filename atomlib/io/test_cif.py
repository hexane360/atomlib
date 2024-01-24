
import textwrap
from io import StringIO
from .cif import CifReader

def test_reader():
    s = """
    1 tis a sentence

      # 4 comments
    # 5 and comments
      6
    """

    r = CifReader(StringIO(s))
    assert r.next_word() == "1"
    assert r.after_eol() == False
    r.next_word(); r.next_word(); r.next_word()
    assert r.after_eol() == True
    assert r.next_word() == "6"
    assert r.line == 6

def test_integration():
    s = textwrap.dedent("""
    data_dataname
    _text_block
    ;
    This is a text block; which ' can ; contain " many characters
    _ - loop_ data_data
    ;
    _quoted_text 'this is a quote'd text" block
    which extends over multiple lines'

    _bare_text bar'etextwith"quotes
    _int 2048
    _float 0.
    _float2 3.5e+5
    _float3 .5(3)
    data_data2
    _tag1
    ; foo
    bar
    ;
    """)

    reader = CifReader(StringIO(s))
    [c1, c2] = reader.parse()
    [d1, d2] = [c1.data_dict, c2.data_dict]

    assert c1.name == 'dataname'
    assert d1['text_block'] == textwrap.dedent("""\
    This is a text block; which ' can ; contain " many characters
    _ - loop_ data_data""")
    assert d1['quoted_text'] == 'this is a quote\'d text" block\n' \
    'which extends over multiple lines'
    assert d1['bare_text'] == "bar'etextwith\"quotes"
    assert d1['int'] == 2048
    assert d1['float'] == 0.
    assert d1['float2'] == 3.5e+5
    assert d1['float3'] == .5

    assert c2.name == 'data2'
    assert d2['tag1'] == 'foo\nbar'