import plotly.express as px
import pandas as pd
import pickle


def visualize_field_hierarchy_nested(field_hierarchy):
    """
    Visualize the hierarchical field data as a nested interactive treemap.

    Parameters:
        field_hierarchy (dict): Hierarchical data dictionary containing fields, subfields, and paper counts.

    Returns:
        None: Displays the treemap visualization.
    """
    # Prepare data for visualization
    data = []
    for field, field_data in field_hierarchy.items():
        field_paper_count = field_data["number_of_papers"]
        subfields = field_data["subfields"]

        # # Add field-level data
        # data.append({
        #     "Level": "Field",
        #     "Name": field,
        #     "Parent": "",  # Top-level has no parent
        #     "Number of Papers": field_paper_count
        # })

        # Add subfield-level data
        for subfield, subfield_data in subfields.items():
            subfield_paper_count = subfield_data["number_of_papers"]
            data.append({
                "Level": "Subfield",
                "Name": subfield,
                "Parent": field,  # Subfield's parent is the Field
                "Number of Papers": subfield_paper_count
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create Treemap with proper nesting
    fig = px.treemap(df,
                     path=["Parent", "Name"],  # Define the hierarchical path
                     values="Number of Papers",  # Size of each node
                     title="Hierarchical Field and Subfield Visualization",
                     color="Number of Papers",  # Color nodes by number of papers
                     hover_data=["Number of Papers"])  # Show number of papers on hover

    # Adjust layout
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    # save to html
    fig.write_html("../data/llm_outputs/statistics/field_hierarchy_treemap.html")
    # Display the treemap
    fig.show()


# Example usage
# Load the hierarchical dictionary from file
def read_hierarchy_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)



if __name__ == "__main__":
    # Load the hierarchical field data from file
    field_hierarchy = read_hierarchy_from_file("../data/llm_outputs/statistics/field_hierarchy_basic_temp.pkl")
    visualize_field_hierarchy_nested(field_hierarchy)

