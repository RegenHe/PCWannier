from PCWannier import IncarParser

class TestIncarParser:
    def test_incar_parser(self):
        parser_data = IncarParser.IncarParser("examples/incar")
        print(parser_data.parse_file())

if __name__ == "__main__":
    test = TestIncarParser()
    test.test_incar_parser()