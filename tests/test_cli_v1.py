import unittest


class CLIV1Tests(unittest.TestCase):
    def test_analysis_stage_selection_and_cutoff_are_parsed(self):
        from coldmd.cli import build_parser

        arguments = build_parser().parse_args(
            [
                "analyze",
                "run",
                "--stage",
                "initial_300K",
                "--stage",
                "compression_10GPa",
                "--si-o-cutoff-A",
                "2.15",
            ]
        )
        self.assertEqual(
            arguments.stages,
            ["initial_300K", "compression_10GPa"],
        )
        self.assertFalse(arguments.all_stages)
        self.assertEqual(arguments.si_o_cutoff_A, 2.15)

    def test_analysis_stage_and_all_stages_are_mutually_exclusive(self):
        from coldmd.cli import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["analyze", "run", "--stage", "initial", "--all-stages"]
            )

    def test_analysis_cutoff_must_be_positive(self):
        from coldmd.cli import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["analyze", "run", "--si-o-cutoff-A", "0"]
            )


if __name__ == "__main__":
    unittest.main()
