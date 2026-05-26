from collections import defaultdict

class MemoryCompressor:
    def __init__(self):
        pass

    def compress(self, messages):
        """
        Converts raw conversation messages into structured memory.
        """

        facts = defaultdict(list)

        for m in messages:
            role = m["role"]
            content = m["content"]

            # simple heuristic classification
            if role == "user":
                facts["user_inputs"].append(content)
            else:
                facts["assistant_outputs"].append(content)

        # build compressed memory
        summary = []

        if facts["user_inputs"]:
            summary.append("User behavior / statements:")
            for item in facts["user_inputs"][-5:]:
                summary.append(f"- {item}")

        if facts["assistant_outputs"]:
            summary.append("Assistant responses context:")
            for item in facts["assistant_outputs"][-5:]:
                summary.append(f"- {item}")

        return "\n".join(summary)