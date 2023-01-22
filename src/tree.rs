use serde::{Serialize, Deserialize};
use std::ops::Deref;
use std::ops::DerefMut;
use std::fmt;

///Node data D, edge data E
#[derive(Serialize, Deserialize)]
pub struct TreeNode<D, E> {
    pub data : D,
    pub maybe_expanded_edges : Option<ExpandedEdges<E>>,
}

impl <D, E> TreeNode<D, E> {
    pub fn has_expanded_children(&self) -> bool {
        self.maybe_expanded_edges.is_some()
    }
}

#[derive(Serialize, Deserialize)]
pub struct ExpandedEdges<E> {
    pub children_start_index : usize,
    pub edges : Vec<E>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NodeIndex {
    index : usize,
}

impl From<usize> for NodeIndex {
    fn from(index : usize) -> Self {
        Self {
            index
        }
    }
}

impl fmt::Display for NodeIndex {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.index)
    }
}

#[derive(Clone, Copy)]
pub struct EdgeIndex {
    starting_node_index : usize,
    children_start_index : usize,
    edge_relative_index : usize,
}

impl EdgeIndex {
    pub fn get_starting_node_index(&self) -> NodeIndex {
        NodeIndex {
            index : self.starting_node_index,         
        }
    }
    pub fn get_edge_relative_index(&self) -> usize {
        self.edge_relative_index
    }
    pub fn get_ending_node_index(&self) -> NodeIndex {
        let index = self.children_start_index + self.edge_relative_index;
        NodeIndex {
            index,
        }
    }
}

///Base data B, node data D, edge data E
#[derive(Serialize, Deserialize)]
pub struct Tree<B, D, E> {
    pub base_data : B,
    pub nodes : Vec<TreeNode<D, E>>,
}

///Traverser-maintained state S
#[derive(Clone)]
pub struct TreeTraverser<S> {
    pub state : S,
    pub index_stack : Vec<NodeIndex>,
}

pub trait TraverserLike {
    fn get_current_node_index(&self) -> NodeIndex;
    fn get_child_edge_indices(&self) -> Vec<EdgeIndex>;
    /// Depth of 1 for root node, +1 each link down
    fn get_depth(&self) -> usize;

    /// Doesn't update the state
    fn go_to_parent_keep_state(&mut self);

    fn has_expanded_children(&self) -> bool {
        self.get_num_children() > 0
    }
    fn get_num_children(&self) -> usize {
        self.get_child_edge_indices().len()
    }
}

pub trait HasTreeWithTraverser {
    type TraverserState;
    type BaseData;
    type NodeData;
    type EdgeData;
    fn get(&self) -> &TreeWithTraverser<Self::TraverserState, Self::BaseData, 
                                        Self::NodeData, Self::EdgeData>;
    fn get_mut(&mut self) -> &mut TreeWithTraverser<Self::TraverserState, Self::BaseData, 
                                                    Self::NodeData, Self::EdgeData>;
    fn get_tree(&self) -> &Tree<Self::BaseData, Self::NodeData, Self::EdgeData> {
        self.get().get_tree()
    }
    fn get_tree_mut(&mut self) -> &mut Tree<Self::BaseData, Self::NodeData, Self::EdgeData> {
        self.get_mut().get_tree_mut()
    }
}

impl <T : HasTreeWithTraverser> TraverserLike for T {
    fn get_current_node_index(&self) -> NodeIndex {
        self.get().get_current_node_index()
    }
    fn get_child_edge_indices(&self) -> Vec<EdgeIndex> {
        self.get().get_child_edge_indices()
    }
    fn get_depth(&self) -> usize {
        self.get().get_depth()
    }
    fn go_to_parent_keep_state(&mut self) {
        self.get_mut().go_to_parent_keep_state();
    }
}

pub struct TreeWithTraverser<S, B, D, E> {
    tree : Tree<B, D, E>,
    traverser : TreeTraverser<S>,
}

impl <S, B, D, E> TraverserLike for TreeWithTraverser<S, B, D, E> {
    fn get_current_node_index(&self) -> NodeIndex {
        self.traverser.get_current_node_index()
    }
    fn get_child_edge_indices(&self) -> Vec<EdgeIndex> {
        self.traverser.get_child_edge_indices(&self.tree).collect()
    }
    fn get_depth(&self) -> usize {
        self.traverser.get_depth()
    }
    fn go_to_parent_keep_state(&mut self) {
        self.traverser.go_to_parent_keep_state()
    }
}

impl <S, B, D, E> TreeWithTraverser<S, B, D, E> {
    pub fn new(base_data : B, init_node_data : D, traverser_state : S) -> Self {
        let tree = Tree::new(base_data, init_node_data);
        let traverser = tree.traverse_from_root(traverser_state);
        Self {
            tree,
            traverser,
        }
    }

    pub fn drain_indices_to_root(&mut self, root_state : S) -> Vec<NodeIndex> {
        self.traverser.drain_indices_to_root(root_state)
    }
    pub fn get_root_data(&self) -> &B {
        &self.get_tree().base_data
    }
    pub fn get_tree(&self) -> &Tree<B, D, E> {
        &self.tree
    }
    pub fn get_tree_mut(&mut self) -> &mut Tree<B, D, E> {
        &mut self.tree
    }
    pub fn add_children(&mut self, children : impl IntoIterator<Item = (E, D)>) {
        self.traverser.add_children(&mut self.tree, children);
    }
    pub fn current_node_mut(&mut self) -> &mut TreeNode<D, E> {
        self.traverser.current_node_mut(&mut self.tree)
    }
    pub fn current_node(&self) -> &TreeNode<D, E> {
        self.traverser.current_node(&self.tree)
    }
    pub fn get_traverser_state(&self) -> &S {
        self.traverser.get_state()
    }
    pub fn manual_move(&mut self, edge_index : EdgeIndex, updated_state : S) {
        self.traverser.manual_move(edge_index, updated_state);
    }
    pub fn get_edge_data(&self, edge_index : EdgeIndex) -> &E {
        self.traverser.get_edge_data(&self.tree, edge_index)
    }
    pub fn get_edge_data_mut(&mut self, edge_index : EdgeIndex) -> &mut E {
        self.traverser.get_edge_data_mut(&mut self.tree, edge_index)
    }
}

impl <B, D, E> Tree<B, D, E> {
    pub fn new(base_data : B, init_node_data : D) -> Tree<B, D, E> {
        let nodes = vec![TreeNode {
            data : init_node_data,
            maybe_expanded_edges : Option::None,
        }];
        Tree {
            base_data,
            nodes
        }
    }
    pub fn get_node_data_for(&mut self, node_index : NodeIndex) -> &mut D {
        &mut self.nodes[node_index.index].data
    }
    pub fn with_traverser_from_root<S>(self, state : S)
           -> TreeWithTraverser<S, B, D, E> {
        let traverser = self.traverse_from_root(state);
        TreeWithTraverser {
            tree : self,
            traverser,
        }
    }
    pub fn traverse_from_root<S>(&self, state : S) -> TreeTraverser<S> {
        let index_stack = vec![NodeIndex {
            index : 0,
        }];
        TreeTraverser {
            state,
            index_stack,
        }
    }
}

impl <S> TreeTraverser<S> {
    pub fn go_to_parent_keep_state(&mut self) {
        self.index_stack.pop();
    }
    pub fn map<T>(self, func : impl FnOnce(S) -> T) -> TreeTraverser<T> {
        TreeTraverser {
            state : func(self.state),
            index_stack : self.index_stack,
        }
    }
    ///Assuming that the current node doesn't currently have any children,
    ///adds the specified children to the tree.
    pub fn add_children<B, D, E>(&self, 
        tree : &mut Tree<B, D, E>,
                                 children : impl IntoIterator<Item = (E, D)>) {
        let children_start_index = tree.nodes.len();

        let mut edges = Vec::new();
        for (edge, data) in children {
            let node = TreeNode {
                data,
                maybe_expanded_edges : Option::None,
            };
            tree.nodes.push(node);
            edges.push(edge);
        }

        let current_node = self.current_node_mut(tree);
        current_node.maybe_expanded_edges = Option::Some(ExpandedEdges {
            children_start_index,
            edges,
        });
    }
    ///Gets the indices of parents all the way to the root, and adjusts
    ///the state of the traverser to the given state for the root of the tree
    pub fn drain_indices_to_root(&mut self, root_state : S) -> Vec<NodeIndex> {
        self.state = root_state;
        let nodes = self.index_stack.drain(..).rev().collect();
        self.index_stack = vec![NodeIndex::from(0)];
        nodes
    }
    pub fn get_depth(&self) -> usize {
        self.index_stack.len()
    }

    fn current_node_index(&self) -> usize {
        self.get_current_node_index().index
    }

    pub fn get_current_node_index(&self) -> NodeIndex {
        self.index_stack[self.index_stack.len() - 1]
    }
    pub fn current_node_mut<'a, B, D, E>(&self, 
                                         tree : &'a mut Tree<B, D, E>) -> &'a mut TreeNode<D, E> {
        &mut tree.nodes[self.current_node_index()]
    }

    pub fn current_node<'a, B, D, E>(&self, 
                          tree : &'a Tree<B, D, E>) -> &'a TreeNode<D, E> {
        &tree.nodes[self.current_node_index()]
    }
    pub fn current_node_data<'a, B, D, E>(&self, 
                                                                    tree : &'a Tree<B, D, E>) -> &'a D {
        &self.current_node(tree).data
    }
    pub fn current_node_data_mut<'a, B, D, E>(&self, 
                                tree : &'a mut Tree<B, D, E>) -> &'a mut D {
        &mut self.current_node_mut(tree).data
    }

    pub fn get_state(&self) -> &S {
        &self.state
    }
    pub fn manual_move(&mut self, edge_index : EdgeIndex, updated_state : S) {
        let child_index = edge_index.get_ending_node_index();
        self.index_stack.push(child_index);
        self.state = updated_state;
    }
    pub fn get_child_edge_indices<'a, B, D, E>(&'a self, 
                                              tree : &'a Tree<B, D, E>) 
        -> impl Iterator<Item = EdgeIndex> + 'a {
        let starting_node_index = self.current_node_index();
        let maybe_expanded_edges = self.current_node(tree).maybe_expanded_edges.as_ref();
        maybe_expanded_edges.into_iter().flat_map(move |expanded_edges| {
            let children_start_index = expanded_edges.children_start_index;
            let num_children = expanded_edges.edges.len(); 
            (0..num_children).map(move |edge_relative_index| {
                EdgeIndex {
                    starting_node_index,
                    children_start_index,
                    edge_relative_index,  
                }
            })
        })
    }

    pub fn get_edge_data<'a, B, D, E>(&self, 
        tree : &'a Tree<B, D, E>,
                                      edge_index : EdgeIndex) -> &'a E {
        let expanded_edges = self.current_node(tree).maybe_expanded_edges.as_ref().unwrap();     
        &expanded_edges.edges[edge_index.edge_relative_index]
    }

    pub fn get_edge_data_mut<'a, B, D, E>(&mut self, 
        tree : &'a mut Tree<B, D, E>,
                                      edge_index : EdgeIndex) -> &'a mut E {
        let expanded_edges = self.current_node_mut(tree).maybe_expanded_edges.as_mut().unwrap();     
        &mut expanded_edges.edges[edge_index.edge_relative_index]
    }
}
