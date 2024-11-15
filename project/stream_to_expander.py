# stream_to_expander.py

import re
import streamlit as st
import sys

# StreamToExpander class to capture agent outputs
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.agent_outputs = {}
        self.current_agent = None
        self.agent_emojis = {
            "Chemistry Expert": "🔬",
            "Molecule Generator": "🧪",
            "Molecule Editor": "⚗️",
            "Crew Manager": "👨‍💼",
        }

    def write(self, data):
        # Write to the terminal
        sys.__stdout__.write(data)
        sys.__stdout__.flush()

        # Filter out unwanted logs and ANSI escape codes
        unwanted_logs = [
            'Checking env var',
            'Mapped key name',
            'Examining the path of torch.classes',
            'LangChainDeprecationWarning',
            'WARNING:',
            'INFO:',
            'DEBUG:',
        ]
        if any(log in data for log in unwanted_logs):
            return  # Skip unwanted logs

        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Split the cleaned data into lines
        lines = cleaned_data.split('\n')
        for line in lines:
            if line.strip() == '':
                continue  # Skip empty lines

            # Detect agent
            agent_match = re.match(r'# Agent: (.+)', line)
            if agent_match:
                self.current_agent = agent_match.group(1).strip()
                if self.current_agent not in self.agent_outputs:
                    self.agent_outputs[self.current_agent] = []
                continue

            # Collect agent output
            if self.current_agent:
                self.agent_outputs[self.current_agent].append(line)
            else:
                # If no current agent, print directly
                with self.expander:
                    st.markdown(line)

        # Update the expander
        self.update_expander()

    def update_expander(self):
        with self.expander:
            # Build the content
            content = ""
            for agent_name, output_lines in self.agent_outputs.items():
                emoji = self.agent_emojis.get(agent_name, "🤖")
                content += f"### {emoji} {agent_name}\n"
                formatted_output = '\n'.join(output_lines)
                content += formatted_output + "\n"
                content += "---\n"  # Separator between agents

            # Update the content
            st.markdown(content)

    def flush(self):
        pass  # Required for file-like objects
