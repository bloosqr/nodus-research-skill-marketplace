"""Where a dipeptide block may be USED, which is not the same question as what it is FOR.

The generator lives at tools/chain-program.py, outside the plugin, but what it encodes is a
chemical claim and these are the cases that claim got wrong at first:

  * A "c-terminal" block answers a hazard that only exists at the C-terminus (diketopiperazine
    needs the free amine of residue 2 to reach the ester holding residue 1 to the resin), so it is
    useful only there.
  * An "anywhere" block answers a hazard that travels with the residue pair (an aspartimide closes
    between the Asp side-chain ester and the very next backbone N-H), so it is useful at EVERY
    position - the C-terminus included, where it is loaded onto the support instead of coupled.

Requiring a block's `where` to equal the position being filled conflated the two and left a
C-terminal Asp-Gly with no remedy at all. And when two blocks want the same residue, one loses -
which is fine, but it must be said out loud rather than vanishing from the route.
"""
import importlib.util
import os
import unittest

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

GENERATOR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "chain-program.py")


def load_generator():
    spec = importlib.util.spec_from_file_location("chain_program", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


class BlockPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = load_generator()
        cls.blocks = cls.cp.monomers()

    def both_routes(self, sequence):
        """The same sequence stepwise and with dipeptides. Returns (stepwise, dipeptide)."""
        plain = self.cp.run(sequence, self.blocks, 60.0, verbose=False, dipeptides=False)
        withdp = self.cp.run(sequence, self.blocks, 60.0, verbose=False, dipeptides=True)
        self.assertTrue(plain["ok"], plain.get("message"))
        self.assertTrue(withdp["ok"], withdp.get("message"))
        return plain, withdp

    def test_an_anywhere_block_is_loaded_at_the_c_terminus(self):
        # Val-Ala-Asp-Gly puts the Asp-Gly at the very end. The aspartimide risk is identical to
        # the mid-chain case, so the block must still be used - by loading, since the C-terminal
        # residue is esterified to the support rather than coupled onto something.
        plain, withdp = self.both_routes(["Val", "Ala", "Asp", "Gly"])
        self.assertEqual(canonical(plain["product"]), canonical(withdp["product"]))
        self.assertLess(len(withdp["rows"]), len(plain["rows"]))
        self.assertTrue(withdp["conserved"])
        self.assertIn("dmb-removal", [row["transform"] for row in withdp["rows"]])
        _, lines, _ = self.cp.program(["Val", "Ala", "Asp", "Gly"], self.blocks, dipeptides=True)
        self.assertIn("Fmoc-Asp(OtBu)-(Dmb)Gly-OH", lines[1])
        self.assertIn("esterified to the support", lines[1])

    def test_a_c_terminal_block_is_not_used_in_the_interior(self):
        # The control that keeps the fix honest: Gly-Pro three residues in is unremarkable, and a
        # generator that used the block "anywhere" would be cargo-culting the remedy past its reason.
        plain, withdp = self.both_routes(["Ala", "Gly", "Pro", "Val"])
        self.assertEqual(len(plain["rows"]), len(withdp["rows"]))
        self.assertEqual(canonical(plain["product"]), canonical(withdp["product"]))

    def test_an_overlapping_pair_that_loses_is_named_in_the_program(self):
        # Val-Asp-Gly-Pro: the Asp-Gly and the Gly-Pro both want the glycine. The c-terminal block
        # takes it - diketopiperazine ends the synthesis where an aspartimide only erodes it - and
        # the route must SAY that the Asp-Gly is now unaddressed.
        _, lines, _ = self.cp.program(["Val", "Asp", "Gly", "Pro"], self.blocks, dipeptides=True)
        notes = [line for line in lines if "UNADDRESSED" in line]
        self.assertEqual(len(notes), 1, lines)
        self.assertIn("Asp-Gly", notes[0])
        self.assertIn("another block", notes[0])
        # and the winning block is the c-terminal one
        self.assertIn("Fmoc-Gly-Pro-OH", "\n".join(lines))

    def test_no_spurious_warning_when_every_pair_is_covered(self):
        for sequence in (["Val", "Asp", "Gly", "Ala"], ["Ala", "Val", "Gly", "Pro"],
                         ["Val", "Ala", "Asp", "Gly"]):
            _, lines, _ = self.cp.program(sequence, self.blocks, dipeptides=True)
            self.assertEqual([line for line in lines if "UNADDRESSED" in line], [], sequence)

    def test_repeated_pairs_still_balance_and_reach_the_same_product(self):
        # Three Asp-Gly pairs means three Dmb groups and ONE acid treatment that takes all of them;
        # the reagent accounting has to scale with appliedTimes or the balance quietly breaks.
        sequence = ["Val", "Asp", "Gly", "Ala", "Asp", "Gly", "Val", "Asp", "Gly"]
        plain, withdp = self.both_routes(sequence)
        self.assertEqual(canonical(plain["product"]), canonical(withdp["product"]))
        self.assertTrue(withdp["conserved"], withdp.get("difference"))
        self.assertEqual([row["transform"] for row in withdp["rows"]].count("dmb-removal"), 1)


if __name__ == "__main__":
    unittest.main()
