import { useReactFlow, type NodeProps } from '@xyflow/react';
import { Play, Square, Zap } from 'lucide-react';
import { CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider } from '@/components/ui/tooltip';
import { useFlowContext } from '@/contexts/flow-context';
import { useLayoutContext } from '@/contexts/layout-context';
import { useNodeContext } from '@/contexts/node-context';
import { useFlowConnection } from '@/hooks/use-flow-connection';
import { formatKeyboardShortcut } from '@/lib/utils';
import { type MacroNewsStartNode } from '../types';
import { NodeShell } from './node-shell';
import { useEffect } from 'react';

export function MacroNewsStartNode({
  data,
  selected,
  id,
  isConnectable,
}: NodeProps<MacroNewsStartNode>) {
  const { currentFlowId } = useFlowContext();
  const nodeContext = useNodeContext();
  const { getAllAgentModels } = nodeContext;
  const { getNodes, getEdges } = useReactFlow();
  const { expandBottomPanel, setBottomPanelTab } = useLayoutContext();

  const flowId = currentFlowId?.toString() || null;
  const {
    isConnecting,
    isConnected,
    isProcessing,
    canRun,
    runFlow,
    stopFlow,
    recoverFlowState
  } = useFlowConnection(flowId);

  // Recover flow state when component mounts or flow changes
  useEffect(() => {
    if (flowId) {
      recoverFlowState();
    }
  }, [flowId, recoverFlowState]);

  const handleStop = () => {
    stopFlow();
  };

  const handlePlay = () => {
    // Show output panel
    expandBottomPanel();
    setBottomPanelTab('output');

    // Graph snapshot
    const allNodes = getNodes();
    const allEdges = getEdges();

    // DFS reachability from this start node
    const reachableNodes = new Set<string>();
    const visited = new Set<string>();
    const dfs = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      if (nodeId !== id) {
        reachableNodes.add(nodeId);
      }
      const outgoingEdges = allEdges.filter(edge => edge.source === nodeId);
      for (const edge of outgoingEdges) {
        dfs(edge.target);
      }
    };
    dfs(id);

    // Include this start node + reachable downstream nodes
    const startNode = allNodes.find(node => node.id === id);
    const agentNodes = [ ...(startNode ? [startNode] : []), ...allNodes.filter(node => reachableNodes.has(node.id)) ];
    const reachableNodeIds = new Set([id, ...reachableNodes]);
    const validEdges = allEdges.filter(edge =>
      reachableNodeIds.has(edge.source) && reachableNodeIds.has(edge.target)
    );

    // Collect agent models
    const agentModels = [];
    const allAgentModels = getAllAgentModels(flowId);
    for (const node of agentNodes) {
      const model = allAgentModels[node.id];
      if (model) {
        agentModels.push({
          agent_id: node.id,
          model_name: model.model_name,
          model_provider: model.provider as any
        });
      }
    }

    // Trigger run with empty tickers; macro agent will drive the flow
    runFlow({
      tickers: [],
      graph_nodes: agentNodes.map(node => ({
        id: node.id,
        type: node.type,
        data: node.data,
        position: node.position
      })),
      graph_edges: validEdges,
      agent_models: agentModels,
      model_name: undefined,
      model_provider: undefined,
    });
  };

  const showAsProcessing = isConnecting || isConnected || isProcessing;

  return (
    <TooltipProvider>
      <NodeShell
        id={id}
        selected={selected}
        isConnectable={isConnectable}
        icon={<Zap className="h-5 w-5" />}
        name={data.name || "Macro News Opportunities"}
        description={data.description}
        hasLeftHandle={false}
        width="w-72"
      >
        <CardContent className="p-0">
          <div className="border-t border-border p-3">
            <div className="flex flex-col gap-3">
              <div className="text-subtitle text-primary">Run</div>
              <div className="flex gap-2">
                <Tooltip delayDuration={200}>
                  <Button
                    size="icon"
                    variant="secondary"
                    className="flex-shrink-0 transition-all duration-200 hover:bg-primary hover:text-primary-foreground active:scale-95"
                    title={showAsProcessing ? "Stop" : `Run (${formatKeyboardShortcut('↵')})`}
                    onClick={showAsProcessing ? handleStop : handlePlay}
                    disabled={!canRun && !showAsProcessing}
                  >
                    {showAsProcessing ? (
                      <Square className="h-3.5 w-3.5" />
                    ) : (
                      <Play className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <TooltipContent side="right">
                    Start macro-driven idea generation
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </CardContent>
      </NodeShell>
    </TooltipProvider>
  );
}

